package dailybee

import (
	"bytes"
	"context"
	"net/url"
	"sort"
	"strings"
	"time"

	"golang.org/x/net/html"

	"github.com/nkanaev/yarr/src/storage/model"
)

// Video is one upload in the bulletin.
type Video struct {
	ItemID      int64         `json:"item_id"`
	VideoID     string        `json:"video_id"`
	Title       string        `json:"title"`
	URL         string        `json:"url"`
	Thumbnail   string        `json:"thumbnail"`
	Summary     string        `json:"summary"`     // first lines of the description
	Description string        `json:"description"` // full description, plain text
	Published   time.Time     `json:"published"`
	Watched     bool          `json:"watched"`
	Saved       bool          `json:"saved"`
	IsShort     bool          `json:"is_short"`
	Live        string        `json:"live,omitempty"` // "live" / "upcoming"
	Duration    time.Duration `json:"duration_seconds"`
	Views       int64         `json:"views"`

	feedID int64
}

// Channel groups a channel's uploads within the bulletin window.
type ChannelGroup struct {
	FeedID int64   `json:"feed_id"`
	Title  string  `json:"title"`
	URL    string  `json:"url"`
	Icon   string  `json:"icon,omitempty"` // data: URI
	Videos []Video `json:"videos"`
}

// Bulletin is everything new across the user's subscriptions.
type Bulletin struct {
	GeneratedAt   time.Time      `json:"generated_at"`
	Since         time.Time      `json:"since"`
	Hours         int            `json:"hours"`
	Channels      []ChannelGroup `json:"channels"`
	TotalVideos   int            `json:"total_videos"`
	Unwatched     int            `json:"unwatched"`
	TotalDuration time.Duration  `json:"total_duration_seconds"`
	HiddenShorts  int            `json:"hidden_shorts"`
	FeedCount     int            `json:"feed_count"` // subscribed channels being watched
}

// BuildBulletin collects the uploads published in the last `hours` hours from
// every YouTube channel feed in the reader.
func (s *Service) BuildBulletin(ctx context.Context, hours int, hideShorts bool) Bulletin {
	if hours <= 0 {
		hours = 24
	}
	now := time.Now()
	b := Bulletin{GeneratedAt: now, Since: now.Add(-time.Duration(hours) * time.Hour), Hours: hours}

	feeds := make(map[int64]model.Feed)
	for _, feed := range s.DB.ListFeeds() {
		if ChannelIDFromFeedURL(feed.FeedLink) != "" {
			feeds[feed.Id] = feed
		}
	}
	b.FeedCount = len(feeds)
	if len(feeds) == 0 {
		return b
	}

	// Page through items newest-first until we fall out of the window.
	var videos []Video
	var after *int64
	const pageSize = 250
scan:
	for {
		items := s.DB.ListItems(model.ItemFilter{After: after}, pageSize, true, true)
		for _, item := range items {
			if item.Date.Before(b.Since) {
				break scan
			}
			if _, ok := feeds[item.FeedId]; !ok {
				continue
			}
			videos = append(videos, videoFromItem(item))
		}
		if len(items) < pageSize {
			break
		}
		last := items[len(items)-1].Id
		after = &last
	}

	ids := make([]string, 0, len(videos))
	for _, v := range videos {
		if v.VideoID != "" {
			ids = append(ids, v.VideoID)
		}
	}
	details := s.videoDetails(ctx, s.Config().APIKey, ids)

	groups := make(map[int64]*ChannelGroup)
	for _, v := range videos {
		if d, ok := details[v.VideoID]; ok {
			v.Duration = d.Duration
			v.Views = d.Views
			if d.LiveBroadcastContent == "live" || d.LiveBroadcastContent == "upcoming" {
				v.Live = d.LiveBroadcastContent
			}
			if d.Duration > 0 && d.Duration <= time.Minute {
				v.IsShort = true
			}
		}
		if hideShorts && v.IsShort {
			b.HiddenShorts++
			continue
		}
		feedID := v.feedID
		g, ok := groups[feedID]
		if !ok {
			feed := feeds[feedID]
			g = &ChannelGroup{
				FeedID: feedID,
				Title:  feed.Title,
				URL:    feed.Link,
			}
			if feed.Icon != nil {
				g.Icon = feed.Icon.DataURI()
			}
			groups[feedID] = g
		}
		g.Videos = append(g.Videos, v)
		b.TotalVideos++
		b.TotalDuration += v.Duration
		if !v.Watched {
			b.Unwatched++
		}
	}

	for _, g := range groups {
		b.Channels = append(b.Channels, *g)
	}
	// Channels with the freshest upload first; videos are already newest-first.
	sort.SliceStable(b.Channels, func(i, j int) bool {
		return b.Channels[i].Videos[0].Published.After(b.Channels[j].Videos[0].Published)
	})
	return b
}

func videoFromItem(item model.Item) Video {
	id := VideoIDFromItem(item.GUID, item.Link)
	v := Video{
		ItemID:    item.Id,
		VideoID:   id,
		Title:     item.Title,
		URL:       item.Link,
		Published: item.Date,
		Watched:   item.Status != model.UNREAD,
		Saved:     item.Status == model.STARRED,
		IsShort:   strings.Contains(item.Link, "/shorts/"),
		feedID:    item.FeedId,
	}
	if id != "" {
		// 16:9 thumbnail (the feed's own hqdefault is letterboxed 4:3)
		v.Thumbnail = "https://i.ytimg.com/vi/" + id + "/mqdefault.jpg"
		if v.URL == "" {
			v.URL = "https://www.youtube.com/watch?v=" + id
		}
	} else {
		for _, m := range item.MediaLinks {
			if m.Type == "image" {
				v.Thumbnail = m.URL
				break
			}
		}
	}
	v.Description = htmlToText(item.Content)
	v.Summary = summarize(v.Description, 2, 260)
	return v
}

// VideoIDFromItem gets the video id from a feed entry ("yt:video:<id>") or,
// failing that, from its link.
func VideoIDFromItem(guid, link string) string {
	if id, ok := strings.CutPrefix(guid, "yt:video:"); ok && id != "" {
		return id
	}
	u, err := url.Parse(link)
	if err != nil {
		return ""
	}
	if id := u.Query().Get("v"); id != "" {
		return id
	}
	if id, ok := strings.CutPrefix(u.Path, "/shorts/"); ok {
		return strings.Trim(id, "/")
	}
	if u.Hostname() == "youtu.be" {
		return strings.Trim(u.Path, "/")
	}
	return ""
}

// htmlToText converts the feed's description HTML back to plain text,
// keeping line breaks.
func htmlToText(content string) string {
	z := html.NewTokenizer(strings.NewReader(content))
	var buf bytes.Buffer
	for {
		switch z.Next() {
		case html.ErrorToken:
			return strings.TrimSpace(buf.String())
		case html.TextToken:
			buf.Write(z.Text())
		case html.StartTagToken, html.SelfClosingTagToken:
			name, _ := z.TagName()
			if string(name) == "br" || string(name) == "p" {
				buf.WriteByte('\n')
			}
		}
	}
}

// summarize returns the first `lines` non-empty lines, capped at `limit` runes.
func summarize(text string, lines, limit int) string {
	var out []string
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		out = append(out, line)
		if len(out) == lines {
			break
		}
	}
	summary := strings.Join(out, " ")
	runes := []rune(summary)
	if len(runes) > limit {
		cut := limit
		for i := limit - 1; i > limit/2; i-- {
			if runes[i] == ' ' {
				cut = i
				break
			}
		}
		summary = strings.TrimRight(string(runes[:cut]), " ,.;:-") + "…"
	}
	return summary
}
