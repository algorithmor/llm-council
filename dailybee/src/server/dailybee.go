package server

import (
	"context"
	"fmt"
	"html/template"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/nkanaev/yarr/src/dailybee"
)

// DailyBee routes: the bulletin is the landing page; yarr's reader UI moves
// to /reader. Pages redirect to the reader (which hosts the login form) when
// auth is enabled and the visitor is not logged in.

func (s *Server) securePage(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.Auth != nil && !s.Auth.IsAuthenticated(r) {
			http.Redirect(w, r, s.BasePath+"/reader", http.StatusFound)
			return
		}
		next(w, r)
	}
}

type videoView struct {
	dailybee.Video
	HasMore      bool // description is longer than the summary
	DurationText string
	ViewsText    string
	PublishedISO string
}

type channelView struct {
	dailybee.ChannelGroup
	IconURL template.URL // data: URI, trusted (produced from stored bytes)
	Initial string
	Videos  []videoView
}

type windowOption struct {
	Hours    int
	Label    string
	Selected bool
}

func (s *Server) handleBulletin(w http.ResponseWriter, r *http.Request) {
	if s.DailyBee == nil {
		http.Redirect(w, r, s.BasePath+"/reader", http.StatusFound)
		return
	}
	cfg := s.DailyBee.Config()
	query := r.URL.Query()

	hours := cfg.Hours
	if h, err := strconv.Atoi(query.Get("hours")); err == nil && h > 0 && h <= 24*30 {
		hours = h
	}
	hideShorts := cfg.HideShorts
	if v := query.Get("shorts"); v != "" {
		hideShorts = v == "hide"
	}

	ctx, cancel := context.WithTimeout(r.Context(), 20*time.Second)
	defer cancel()
	b := s.DailyBee.BuildBulletin(ctx, hours, hideShorts)

	channels := make([]channelView, 0, len(b.Channels))
	for _, ch := range b.Channels {
		cv := channelView{ChannelGroup: ch, IconURL: template.URL(ch.Icon)}
		for _, r := range strings.TrimSpace(ch.Title) {
			cv.Initial = strings.ToUpper(string(r))
			break
		}
		for _, v := range ch.Videos {
			cv.Videos = append(cv.Videos, videoView{
				Video:        v,
				HasMore:      utf8.RuneCountInString(v.Description) > utf8.RuneCountInString(v.Summary)+3,
				DurationText: dailybee.FormatClock(v.Duration),
				ViewsText:    dailybee.FormatCount(v.Views),
				PublishedISO: v.Published.UTC().Format(time.RFC3339),
			})
		}
		channels = append(channels, cv)
	}

	windows := []windowOption{{24, "24 hours", false}, {48, "2 days", false}, {168, "7 days", false}}
	for i := range windows {
		windows[i].Selected = windows[i].Hours == hours
	}

	var pending int32
	if s.Scheduler != nil {
		pending = s.Scheduler.FeedsPending()
	}

	writeHTML(w, http.StatusOK, s.Template.Lookup("bulletin.html"), map[string]any{
		"base":         s.BasePath,
		"configured":   cfg.Configured(),
		"bulletin":     b,
		"channels":     channels,
		"watchTime":    dailybee.FormatWatchTime(b.TotalDuration),
		"windowLabel":  dailybee.FormatWindow(hours),
		"windows":      windows,
		"hideShorts":   hideShorts,
		"status":       s.DailyBee.Status(),
		"refreshing":   pending > 0,
		"requiresAuth": s.Auth != nil,
	})
}

func (s *Server) handleSetup(w http.ResponseWriter, r *http.Request) {
	if s.DailyBee == nil {
		http.Redirect(w, r, s.BasePath+"/reader", http.StatusFound)
		return
	}
	svc := s.DailyBee
	message, errMessage := "", ""

	switch r.Method {
	case http.MethodGet:
		query := r.URL.Query()
		if e := query.Get("import_error"); e != "" {
			errMessage = "Import failed: " + e
		}
		if n := query.Get("imported"); n != "" {
			message = "Imported " + n + " new channels (of " + query.Get("of") +
				" in the file). New videos are being fetched now."
		}
		if query.Get("saved") != "" {
			st := svc.Status()
			if st.Error != "" {
				errMessage = st.Error
			} else {
				message = "Saved. Found " + strconv.Itoa(st.Channels) + " subscriptions (" +
					strconv.Itoa(st.Added) + " new). New videos are being fetched now."
			}
		}
	case http.MethodPost:
		if err := r.ParseForm(); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		cfg := svc.StoredConfig()
		if key := strings.TrimSpace(r.PostForm.Get("api_key")); key != "" {
			cfg.APIKey = key
		}
		if channel := strings.TrimSpace(r.PostForm.Get("channel")); channel != "" && channel != cfg.Channel {
			cfg.Channel = channel
			cfg.ChannelID = ""
		}
		if h, err := strconv.Atoi(r.PostForm.Get("hours")); err == nil && h > 0 && h <= 24*30 {
			cfg.Hours = h
		}
		cfg.HideShorts = r.PostForm.Get("hide_shorts") != ""
		cfg.Prune = r.PostForm.Get("prune") != ""
		if !svc.SaveConfig(cfg) {
			errMessage = "Could not save settings."
			break
		}
		if svc.Config().Configured() {
			ctx, cancel := context.WithTimeout(r.Context(), 3*time.Minute)
			svc.Sync(ctx)
			cancel()
		}
		// Post/Redirect/Get so a reload doesn't resubmit the form.
		http.Redirect(w, r, s.BasePath+"/setup?saved=1", http.StatusSeeOther)
		return
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	effective := svc.Config()
	writeHTML(w, http.StatusOK, s.Template.Lookup("setup.html"), map[string]any{
		"base":         s.BasePath,
		"config":       effective,
		"hasKey":       effective.APIKey != "",
		"maskedKey":    effective.MaskedAPIKey(),
		"envKey":       svc.EnvAPIKey != "",
		"envChannel":   svc.EnvChannel != "",
		"status":       svc.Status(),
		"message":      message,
		"error":        errMessage,
		"requiresAuth": s.Auth != nil,
	})
}

func (s *Server) handleTakeoutImport(w http.ResponseWriter, r *http.Request) {
	if s.DailyBee == nil || r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	fail := func(msg string) {
		http.Redirect(w, r, s.BasePath+"/setup?import_error="+url.QueryEscape(msg), http.StatusSeeOther)
	}
	file, _, err := r.FormFile("takeout")
	if err != nil {
		fail("choose a subscriptions.csv file first.")
		return
	}
	defer file.Close()
	added, total, err := s.DailyBee.ImportTakeout(file)
	if err != nil {
		fail(err.Error())
		return
	}
	http.Redirect(w, r, fmt.Sprintf("%s/setup?imported=%d&of=%d", s.BasePath, added, total), http.StatusSeeOther)
}

func (s *Server) handleBulletinAPI(w http.ResponseWriter, r *http.Request) {
	if s.DailyBee == nil {
		w.WriteHeader(http.StatusNotFound)
		return
	}
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	cfg := s.DailyBee.Config()
	hours := cfg.Hours
	if h, err := strconv.Atoi(r.URL.Query().Get("hours")); err == nil && h > 0 && h <= 24*30 {
		hours = h
	}
	hideShorts := cfg.HideShorts
	if v := r.URL.Query().Get("shorts"); v != "" {
		hideShorts = v == "hide"
	}
	writeJSON(w, http.StatusOK, s.DailyBee.BuildBulletin(r.Context(), hours, hideShorts))
}

func (s *Server) handleYouTubeSync(w http.ResponseWriter, r *http.Request) {
	if s.DailyBee == nil {
		w.WriteHeader(http.StatusNotFound)
		return
	}
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, http.StatusOK, s.DailyBee.Status())
	case http.MethodPost:
		ctx, cancel := context.WithTimeout(r.Context(), 3*time.Minute)
		defer cancel()
		status, err := s.DailyBee.Sync(ctx)
		if err != nil {
			writeJSON(w, http.StatusBadGateway, map[string]any{"error": err.Error(), "status": status})
			return
		}
		writeJSON(w, http.StatusOK, status)
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
	}
}
