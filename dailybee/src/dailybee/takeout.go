package dailybee

import (
	"encoding/csv"
	"errors"
	"io"
	"strings"

	"github.com/nkanaev/yarr/src/storage/model"
)

// ImportTakeout adds channels from a Google Takeout subscriptions.csv
// ("Channel Id,Channel Url,Channel Title"). It needs no API key, which makes
// it the way in for people who keep their subscriptions private.
func (s *Service) ImportTakeout(r io.Reader) (added, total int, err error) {
	reader := csv.NewReader(r)
	reader.FieldsPerRecord = -1
	reader.TrimLeadingSpace = true
	records, err := reader.ReadAll()
	if err != nil {
		return 0, 0, err
	}

	var channels []Channel
	for _, rec := range records {
		if len(rec) == 0 {
			continue
		}
		id := strings.TrimSpace(strings.TrimPrefix(rec[0], "\ufeff"))
		if !channelIDRe.MatchString(id) {
			continue // header row or junk
		}
		ch := Channel{ID: id, Title: id}
		if len(rec) >= 3 && strings.TrimSpace(rec[2]) != "" {
			ch.Title = strings.TrimSpace(rec[2])
		}
		channels = append(channels, ch)
	}
	if len(channels) == 0 {
		return 0, 0, errors.New("no channels found; expected Google Takeout's subscriptions.csv")
	}

	folderID := s.ensureFolder()
	existing := make(map[string]bool)
	for _, feed := range s.DB.ListFeeds() {
		if id := ChannelIDFromFeedURL(feed.FeedLink); id != "" {
			existing[id] = true
		}
	}
	for _, ch := range channels {
		if existing[ch.ID] {
			continue
		}
		existing[ch.ID] = true
		if s.DB.CreateFeed(model.CreateFeedParams{
			Title:    ch.Title,
			Link:     ch.URL(),
			FeedLink: ch.FeedURL(),
			FolderID: &folderID,
		}) != nil {
			added++
		}
	}
	if s.Refresher != nil {
		s.Refresher.RefreshFeeds()
	}
	return added, len(channels), nil
}
