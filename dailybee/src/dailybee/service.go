package dailybee

import (
	"context"
	"errors"
	"log"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/nkanaev/yarr/src/storage"
	"github.com/nkanaev/yarr/src/storage/model"
)

// FolderTitle is the reader folder that holds the synced YouTube channels.
const FolderTitle = "YouTube Subscriptions"

const configKey = "dailybee_config"

// Config is DailyBee's persisted configuration. It lives in the settings
// table under its own key and is never sent to the browser as-is.
type Config struct {
	APIKey     string `json:"api_key"`
	Channel    string `json:"channel"`     // what the user typed: UC..., @handle or URL
	ChannelID  string `json:"channel_id"`  // resolved UC... id
	Hours      int    `json:"hours"`       // bulletin window
	HideShorts bool   `json:"hide_shorts"` // drop Shorts from the bulletin
	Prune      bool   `json:"prune"`       // remove channels you unsubscribed from
}

func (c Config) Configured() bool { return c.APIKey != "" && c.Channel != "" }

// MaskedAPIKey shows just enough of the key to recognise it.
func (c Config) MaskedAPIKey() string {
	if len(c.APIKey) <= 4 {
		return strings.Repeat("•", len(c.APIKey))
	}
	return "••••••••" + c.APIKey[len(c.APIKey)-4:]
}

// SyncStatus describes the last subscription sync.
type SyncStatus struct {
	Running  bool      `json:"running"`
	LastRun  time.Time `json:"last_run"`
	Error    string    `json:"error,omitempty"`
	Channels int       `json:"channels"`
	Added    int       `json:"added"`
	Removed  int       `json:"removed"`
}

// Refresher is the part of yarr's worker DailyBee needs.
type Refresher interface {
	RefreshFeeds()
}

// Service ties the YouTube API to yarr's storage and feed worker.
type Service struct {
	DB        storage.Storage
	Refresher Refresher

	// Values from the environment take precedence over the stored config.
	EnvAPIKey  string
	EnvChannel string

	syncMu    sync.Mutex
	statusMu  sync.Mutex
	status    SyncStatus
	detailsMu sync.Mutex
	details   map[string]cachedDetails
}

type cachedDetails struct {
	VideoDetails
	fetched time.Time
}

// NewService creates the service, reading DAILYBEE_YOUTUBE_API_KEY and
// DAILYBEE_YOUTUBE_CHANNEL from the environment.
func NewService(db storage.Storage, refresher Refresher) *Service {
	return &Service{
		DB:         db,
		Refresher:  refresher,
		EnvAPIKey:  strings.TrimSpace(os.Getenv("DAILYBEE_YOUTUBE_API_KEY")),
		EnvChannel: strings.TrimSpace(os.Getenv("DAILYBEE_YOUTUBE_CHANNEL")),
		details:    make(map[string]cachedDetails),
	}
}

// StoredConfig returns the config as saved through the UI.
func (s *Service) StoredConfig() Config {
	cfg := Config{Hours: 24}
	s.DB.GetSettingValue(configKey, &cfg)
	if cfg.Hours <= 0 {
		cfg.Hours = 24
	}
	return cfg
}

// Config returns the effective config (environment overrides applied).
func (s *Service) Config() Config {
	cfg := s.StoredConfig()
	if s.EnvAPIKey != "" {
		cfg.APIKey = s.EnvAPIKey
	}
	if s.EnvChannel != "" && s.EnvChannel != cfg.Channel {
		cfg.Channel = s.EnvChannel
		cfg.ChannelID = ""
	}
	return cfg
}

func (s *Service) SaveConfig(cfg Config) bool {
	return s.DB.SetSettingValue(configKey, cfg)
}

func (s *Service) Status() SyncStatus {
	s.statusMu.Lock()
	defer s.statusMu.Unlock()
	return s.status
}

func (s *Service) setStatus(fn func(*SyncStatus)) {
	s.statusMu.Lock()
	defer s.statusMu.Unlock()
	fn(&s.status)
}

// ensureFolder returns the id of the YouTube folder, creating it if needed.
func (s *Service) ensureFolder() int64 {
	for _, f := range s.DB.ListFolders() {
		if f.Title == FolderTitle {
			return f.Id
		}
	}
	return s.DB.CreateFolder(FolderTitle).Id
}

// Sync mirrors the user's YouTube subscriptions into reader feeds and kicks
// off a feed refresh. Only one sync runs at a time.
func (s *Service) Sync(ctx context.Context) (SyncStatus, error) {
	if !s.syncMu.TryLock() {
		return s.Status(), errors.New("a sync is already running")
	}
	defer s.syncMu.Unlock()
	s.setStatus(func(st *SyncStatus) { st.Running = true })

	result, err := s.sync(ctx)
	result.LastRun = time.Now()
	if err != nil {
		result.Error = err.Error()
		log.Printf("dailybee: subscription sync failed: %s", err)
	} else {
		log.Printf("dailybee: synced %d channels (%d added, %d removed)",
			result.Channels, result.Added, result.Removed)
	}
	s.setStatus(func(st *SyncStatus) { *st = result })
	return result, err
}

func (s *Service) sync(ctx context.Context) (SyncStatus, error) {
	var result SyncStatus
	cfg := s.Config()
	if !cfg.Configured() {
		return result, errors.New("add your YouTube API key and channel first")
	}

	channelID := cfg.ChannelID
	if channelID == "" {
		id, err := ResolveChannelID(ctx, cfg.APIKey, cfg.Channel)
		if err != nil {
			return result, err
		}
		channelID = id
		if s.EnvChannel == "" {
			stored := s.StoredConfig()
			stored.ChannelID = id
			s.SaveConfig(stored)
		}
	}

	channels, err := ListSubscriptions(ctx, cfg.APIKey, channelID)
	if err != nil {
		return result, err
	}
	result.Channels = len(channels)

	folderID := s.ensureFolder()
	existing := make(map[string]model.Feed)
	for _, feed := range s.DB.ListFeeds() {
		if id := ChannelIDFromFeedURL(feed.FeedLink); id != "" {
			existing[id] = feed
		}
	}

	subscribed := make(map[string]bool, len(channels))
	var needIcon []iconJob
	for _, ch := range channels {
		subscribed[ch.ID] = true
		feed, ok := existing[ch.ID]
		if !ok {
			created := s.DB.CreateFeed(model.CreateFeedParams{
				Title:       ch.Title,
				Description: ch.Description,
				Link:        ch.URL(),
				FeedLink:    ch.FeedURL(),
				FolderID:    &folderID,
			})
			if created == nil {
				continue
			}
			feed = *created
			result.Added++
		}
		if feed.Icon == nil && ch.Thumbnail != "" {
			needIcon = append(needIcon, iconJob{feed.Id, ch.Thumbnail})
		}
	}

	if cfg.Prune {
		for id, feed := range existing {
			if !subscribed[id] && feed.FolderId != nil && *feed.FolderId == folderID {
				s.DB.DeleteFeed(feed.Id)
				result.Removed++
			}
		}
	}

	// Channel avatars make much nicer icons than youtube.com's favicon.
	s.fetchIcons(ctx, needIcon)

	if s.Refresher != nil {
		s.Refresher.RefreshFeeds()
	}
	return result, nil
}

type iconJob struct {
	feedID int64
	url    string
}

func (s *Service) fetchIcons(ctx context.Context, jobs []iconJob) {
	queue := make(chan iconJob)
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range queue {
				icon, err := fetchImage(ctx, job.url)
				if err != nil || len(icon) == 0 {
					log.Printf("dailybee: channel avatar %s: %v", job.url, err)
					continue
				}
				i := model.Icon(icon)
				s.DB.UpdateFeed(job.feedID, model.UpdateFeedParams{Icon: model.SetNullable(&i)})
			}
		}()
	}
	for _, job := range jobs {
		queue <- job
	}
	close(queue)
	wg.Wait()
}

// StartAutoSync re-syncs subscriptions shortly after start-up and then once a
// day, so newly followed channels show up without any clicking.
func (s *Service) StartAutoSync() {
	go func() {
		time.Sleep(15 * time.Second)
		for {
			if s.Config().Configured() {
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
				s.Sync(ctx)
				cancel()
			}
			time.Sleep(24 * time.Hour)
		}
	}()
}

// videoDetails returns cached durations/views, fetching the missing ones.
// Failures are logged and simply leave the bulletin without that detail.
func (s *Service) videoDetails(ctx context.Context, apiKey string, ids []string) map[string]VideoDetails {
	const ttl = time.Hour
	result := make(map[string]VideoDetails, len(ids))
	var missing []string

	s.detailsMu.Lock()
	now := time.Now()
	for _, id := range ids {
		if d, ok := s.details[id]; ok && now.Sub(d.fetched) < ttl {
			result[id] = d.VideoDetails
		} else {
			missing = append(missing, id)
		}
	}
	s.detailsMu.Unlock()

	if len(missing) == 0 || apiKey == "" {
		return result
	}
	fetched, err := GetVideoDetails(ctx, apiKey, missing)
	if err != nil {
		log.Printf("dailybee: could not fetch video details: %s", err)
	}

	s.detailsMu.Lock()
	defer s.detailsMu.Unlock()
	for id, d := range fetched {
		s.details[id] = cachedDetails{VideoDetails: d, fetched: now}
		result[id] = d
	}
	// keep the cache from growing forever
	for id, d := range s.details {
		if now.Sub(d.fetched) > 24*time.Hour {
			delete(s.details, id)
		}
	}
	return result
}
