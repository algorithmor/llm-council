package dailybee

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/nkanaev/yarr/src/storage"
	"github.com/nkanaev/yarr/src/storage/model"
)

func TestParseISODuration(t *testing.T) {
	cases := map[string]time.Duration{
		"PT15S":    15 * time.Second,
		"PT4M5S":   4*time.Minute + 5*time.Second,
		"PT1H2M3S": time.Hour + 2*time.Minute + 3*time.Second,
		"PT2H":     2 * time.Hour,
		"P1DT1H":   25 * time.Hour,
		"P0D":      0,
		"garbage":  0,
		"":         0,
	}
	for in, want := range cases {
		if got := ParseISODuration(in); got != want {
			t.Errorf("ParseISODuration(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestFormatting(t *testing.T) {
	clock := map[time.Duration]string{
		0:                "",
		65 * time.Second: "1:05",
		time.Hour + 2*time.Minute + 3*time.Second: "1:02:03",
	}
	for in, want := range clock {
		if got := FormatClock(in); got != want {
			t.Errorf("FormatClock(%v) = %q, want %q", in, got, want)
		}
	}
	counts := map[int64]string{0: "", 950: "950", 1000: "1K", 1250: "1.2K", 12_345: "12K", 1_500_000: "1.5M", 25_000_000: "25M"}
	for in, want := range counts {
		if got := FormatCount(in); got != want {
			t.Errorf("FormatCount(%d) = %q, want %q", in, got, want)
		}
	}
	if got := FormatWatchTime(3*time.Hour + 20*time.Minute); got != "3 h 20 min" {
		t.Errorf("FormatWatchTime = %q", got)
	}
	if got := FormatWindow(168); got != "7 days" {
		t.Errorf("FormatWindow(168) = %q", got)
	}
}

func TestVideoIDFromItem(t *testing.T) {
	cases := []struct{ guid, link, want string }{
		{"yt:video:abc123DEF45", "", "abc123DEF45"},
		{"", "https://www.youtube.com/watch?v=xyz", "xyz"},
		{"", "https://www.youtube.com/shorts/shortID", "shortID"},
		{"", "https://youtu.be/beID", "beID"},
		{"", "https://example.com/post", ""},
	}
	for _, c := range cases {
		if got := VideoIDFromItem(c.guid, c.link); got != c.want {
			t.Errorf("VideoIDFromItem(%q, %q) = %q, want %q", c.guid, c.link, got, c.want)
		}
	}
}

func TestChannelIDFromFeedURL(t *testing.T) {
	if got := ChannelIDFromFeedURL(ChannelFeedURL("UCabc")); got != "UCabc" {
		t.Errorf("round trip failed: %q", got)
	}
	if got := ChannelIDFromFeedURL("https://example.com/feeds/videos.xml?channel_id=UCabc"); got != "" {
		t.Errorf("non-youtube host accepted: %q", got)
	}
}

func TestSummarize(t *testing.T) {
	text := "First line.\n\nSecond line here.\nThird line."
	if got := summarize(text, 2, 200); got != "First line. Second line here." {
		t.Errorf("summarize = %q", got)
	}
	long := strings.Repeat("word ", 100)
	got := summarize(long, 2, 50)
	if !strings.HasSuffix(got, "…") || len([]rune(got)) > 51 {
		t.Errorf("summarize long = %q", got)
	}
	if got := htmlToText(`Hello<br>World &amp; <a href="x">link</a>`); got != "Hello\nWorld & link" {
		t.Errorf("htmlToText = %q", got)
	}
}

// fakeYouTube serves the handful of Data API endpoints DailyBee uses.
func fakeYouTube(t *testing.T) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		if r.URL.Path == "/avatar.png" {
			w.Header().Set("Content-Type", "image/png")
			w.Write([]byte("\x89PNG\r\n\x1a\nfake"))
			return
		}
		if q.Get("key") != "good-key" {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"error":{"code":400,"message":"API key not valid. Please pass a valid API key.","errors":[{"reason":"badRequest"}]}}`))
			return
		}
		switch r.URL.Path {
		case "/channels":
			if q.Get("forHandle") == "@mybee" {
				w.Write([]byte(`{"items":[{"id":"UCmemememememememememe00"}]}`))
			} else {
				w.Write([]byte(`{"items":[]}`))
			}
		case "/subscriptions":
			if q.Get("channelId") == "UCprivateprivateprivat00" {
				w.WriteHeader(http.StatusForbidden)
				w.Write([]byte(`{"error":{"code":403,"message":"forbidden","errors":[{"reason":"subscriptionForbidden"}]}}`))
				return
			}
			// two pages of results
			if q.Get("pageToken") == "" {
				w.Write([]byte(`{"nextPageToken":"p2","items":[{"snippet":{"title":"Alpha","resourceId":{"channelId":"UCalphaalphaalphaalpha00"},"thumbnails":{"default":{"url":"` + "http://" + r.Host + `/avatar.png"}}}}]}`))
			} else {
				w.Write([]byte(`{"items":[{"snippet":{"title":"Beta","resourceId":{"channelId":"UCbetabetabetabetabeta00"}}}]}`))
			}
		case "/videos":
			var items []map[string]any
			for _, id := range strings.Split(q.Get("id"), ",") {
				dur := "PT12M30S"
				if id == "short1" {
					dur = "PT45S"
				}
				items = append(items, map[string]any{
					"id":             id,
					"snippet":        map[string]any{"liveBroadcastContent": "none"},
					"contentDetails": map[string]any{"duration": dur},
					"statistics":     map[string]any{"viewCount": "12345"},
				})
			}
			json.NewEncoder(w).Encode(map[string]any{"items": items})
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
}

type countingRefresher struct{ calls int }

func (c *countingRefresher) RefreshFeeds() { c.calls++ }

func newTestService(t *testing.T, apiBase string) (*Service, *countingRefresher, storage.Storage) {
	t.Helper()
	log.SetOutput(io.Discard)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })
	db, err := storage.New(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	old := APIBase
	APIBase = apiBase
	t.Cleanup(func() { APIBase = old })
	ref := &countingRefresher{}
	svc := NewService(db, ref)
	svc.EnvAPIKey, svc.EnvChannel = "", ""
	return svc, ref, db
}

func TestResolveChannelID(t *testing.T) {
	srv := fakeYouTube(t)
	defer srv.Close()
	svc, _, _ := newTestService(t, srv.URL)
	_ = svc
	ctx := context.Background()

	for _, in := range []string{"UCmemememememememememe00", "https://www.youtube.com/channel/UCmemememememememememe00", "@mybee", "mybee", "https://www.youtube.com/@mybee"} {
		id, err := ResolveChannelID(ctx, "good-key", in)
		if err != nil || id != "UCmemememememememememe00" {
			t.Errorf("ResolveChannelID(%q) = %q, %v", in, id, err)
		}
	}
	if _, err := ResolveChannelID(ctx, "good-key", "@nobody"); err == nil {
		t.Error("expected error for unknown handle")
	}
	_, err := ResolveChannelID(ctx, "bad-key", "@mybee")
	var apiErr *APIError
	if !errors.As(err, &apiErr) || !strings.Contains(err.Error(), "not valid") {
		t.Errorf("expected friendly invalid-key error, got %v", err)
	}
}

func TestSyncAndBulletin(t *testing.T) {
	srv := fakeYouTube(t)
	defer srv.Close()
	svc, ref, db := newTestService(t, srv.URL)
	ctx := context.Background()

	if _, err := svc.Sync(ctx); err == nil {
		t.Fatal("sync without config should fail")
	}

	svc.SaveConfig(Config{APIKey: "good-key", Channel: "@mybee", Hours: 24})
	st, err := svc.Sync(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if st.Channels != 2 || st.Added != 2 || ref.calls != 1 {
		t.Fatalf("unexpected sync result %+v (refresh calls %d)", st, ref.calls)
	}
	if svc.StoredConfig().ChannelID != "UCmemememememememememe00" {
		t.Error("resolved channel id was not stored")
	}

	// second sync is idempotent
	if st, _ := svc.Sync(ctx); st.Added != 0 {
		t.Errorf("second sync added %d feeds", st.Added)
	}

	var alpha, beta model.Feed
	for _, f := range db.ListFeeds() {
		switch ChannelIDFromFeedURL(f.FeedLink) {
		case "UCalphaalphaalphaalpha00":
			alpha = f
		case "UCbetabetabetabetabeta00":
			beta = f
		}
	}
	if alpha.Id == 0 || beta.Id == 0 {
		t.Fatal("channel feeds were not created")
	}
	if alpha.Icon == nil {
		t.Error("channel avatar was not stored as the feed icon")
	}

	now := time.Now()
	db.CreateItems([]model.Item{
		{GUID: "yt:video:new1", FeedId: alpha.Id, Title: "Fresh video", Link: "https://www.youtube.com/watch?v=new1",
			Content: "Line one<br>Line two<br>Line three", Date: now.Add(-2 * time.Hour), Status: model.UNREAD},
		{GUID: "yt:video:short1", FeedId: alpha.Id, Title: "A short", Link: "https://www.youtube.com/shorts/short1",
			Date: now.Add(-3 * time.Hour), Status: model.UNREAD},
		{GUID: "yt:video:new2", FeedId: beta.Id, Title: "Beta upload", Link: "https://www.youtube.com/watch?v=new2",
			Date: now.Add(-1 * time.Hour), Status: model.READ},
		{GUID: "yt:video:old1", FeedId: beta.Id, Title: "Old news", Link: "https://www.youtube.com/watch?v=old1",
			Date: now.Add(-30 * time.Hour), Status: model.UNREAD},
	})
	// a non-YouTube feed must not show up
	blog := db.CreateFeed(model.CreateFeedParams{Title: "Blog", FeedLink: "https://example.com/feed.xml"})
	db.CreateItems([]model.Item{{GUID: "b1", FeedId: blog.Id, Title: "Blog post", Date: now.Add(-time.Hour)}})

	b := svc.BuildBulletin(ctx, 24, false)
	if b.TotalVideos != 3 || b.Unwatched != 2 || len(b.Channels) != 2 {
		t.Fatalf("bulletin: videos=%d unwatched=%d channels=%d", b.TotalVideos, b.Unwatched, len(b.Channels))
	}
	if b.Channels[0].Title != "Beta" {
		t.Errorf("channel with the newest upload should come first, got %q", b.Channels[0].Title)
	}
	fresh := b.Channels[1].Videos[0]
	if fresh.VideoID != "new1" || fresh.Duration != 12*time.Minute+30*time.Second || fresh.Views != 12345 {
		t.Errorf("enrichment missing: %+v", fresh)
	}
	if fresh.Summary != "Line one Line two" || !strings.Contains(fresh.Description, "Line three") {
		t.Errorf("summary/description: %q / %q", fresh.Summary, fresh.Description)
	}
	if !b.Channels[1].Videos[1].IsShort {
		t.Error("short was not detected")
	}

	b = svc.BuildBulletin(ctx, 24, true)
	if b.TotalVideos != 2 || b.HiddenShorts != 1 {
		t.Errorf("hide shorts: videos=%d hidden=%d", b.TotalVideos, b.HiddenShorts)
	}
	if b = svc.BuildBulletin(ctx, 48, false); b.TotalVideos != 4 {
		t.Errorf("48h window: videos=%d", b.TotalVideos)
	}
}

func TestSyncPrivateSubscriptions(t *testing.T) {
	srv := fakeYouTube(t)
	defer srv.Close()
	svc, _, _ := newTestService(t, srv.URL)
	svc.SaveConfig(Config{APIKey: "good-key", Channel: "UCprivateprivateprivat00"})
	_, err := svc.Sync(context.Background())
	if err == nil || !strings.Contains(err.Error(), "private") {
		t.Errorf("expected a hint about private subscriptions, got %v", err)
	}
	if svc.Status().Error == "" {
		t.Error("status should record the error")
	}
}

func TestEnvOverrides(t *testing.T) {
	svc, _, _ := newTestService(t, "http://unused")
	svc.SaveConfig(Config{APIKey: "stored", Channel: "@stored", ChannelID: "UCstoredstoredstoredst00"})
	svc.EnvAPIKey, svc.EnvChannel = "env-key", "@env"
	cfg := svc.Config()
	if cfg.APIKey != "env-key" || cfg.Channel != "@env" || cfg.ChannelID != "" {
		t.Errorf("env did not override stored config: %+v", cfg)
	}
	if got := (Config{APIKey: "AIzaSyABCDEFG1234"}).MaskedAPIKey(); strings.Contains(got, "AIza") || !strings.HasSuffix(got, "1234") {
		t.Errorf("MaskedAPIKey leaked the key: %q", got)
	}
}

func TestImportTakeout(t *testing.T) {
	svc, ref, db := newTestService(t, "http://unused")
	csv := "\ufeffChannel Id,Channel Url,Channel Title\n" +
		"UCalphaalphaalphaalpha00,http://www.youtube.com/channel/UCalphaalphaalphaalpha00,Alpha\n" +
		"UCbetabetabetabetabeta00,http://www.youtube.com/channel/UCbetabetabetabetabeta00,\"Beta, Inc\"\n"
	added, total, err := svc.ImportTakeout(strings.NewReader(csv))
	if err != nil || added != 2 || total != 2 || ref.calls != 1 {
		t.Fatalf("import: added=%d total=%d err=%v refresh=%d", added, total, err, ref.calls)
	}
	if added, _, _ := svc.ImportTakeout(strings.NewReader(csv)); added != 0 {
		t.Errorf("re-import added %d", added)
	}
	titles := map[string]bool{}
	for _, f := range db.ListFeeds() {
		titles[f.Title] = true
	}
	if !titles["Beta, Inc"] {
		t.Errorf("quoted title not parsed: %v", titles)
	}
	if _, _, err := svc.ImportTakeout(strings.NewReader("not,a,takeout\n")); err == nil {
		t.Error("expected error for a file without channels")
	}
}
