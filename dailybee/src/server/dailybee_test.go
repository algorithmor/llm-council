package server

import (
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/nkanaev/yarr/src/dailybee"
	"github.com/nkanaev/yarr/src/storage/model"
)

func beeServer(t *testing.T) (*Server, *dailybee.Service) {
	t.Helper()
	log.SetOutput(io.Discard)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })
	srv := testServer()
	db := srv.Storage.GetStorage(nil)
	svc := dailybee.NewService(db, nil)
	svc.EnvAPIKey, svc.EnvChannel = "", ""
	srv.DailyBee = svc
	return srv, svc
}

func get(t *testing.T, h http.Handler, target string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", target, nil))
	return rec
}

func TestBulletinWelcome(t *testing.T) {
	srv, _ := beeServer(t)
	rec := get(t, srv.Handler(), "/")
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), "Welcome to DailyBee") {
		t.Fatalf("status %d, body:\n%s", rec.Code, rec.Body.String())
	}
}

func TestBulletinRendersVideos(t *testing.T) {
	srv, svc := beeServer(t)
	db := srv.Storage.GetStorage(nil)
	svc.SaveConfig(dailybee.Config{APIKey: "k", Channel: "@me", Hours: 24})

	icon := model.Icon([]byte("\x89PNG\r\n\x1a\nfake"))
	feed := db.CreateFeed(model.CreateFeedParams{
		Title:    "Ünicode Channel",
		Link:     "https://www.youtube.com/channel/UCalphaalphaalphaalpha00",
		FeedLink: dailybee.ChannelFeedURL("UCalphaalphaalphaalpha00"),
	})
	db.UpdateFeed(feed.Id, model.UpdateFeedParams{Icon: model.SetNullable(&icon)})
	db.CreateItems([]model.Item{{
		GUID: "yt:video:vid1", FeedId: feed.Id, Title: "Hello <world>",
		Link:    "https://www.youtube.com/watch?v=vid1",
		Content: "Summary line<br>more<br>and a lot more text in the description",
		Date:    time.Now().Add(-time.Hour),
	}})

	rec := get(t, srv.Handler(), "/")
	body := rec.Body.String()
	for _, want := range []string{
		"1 new video</strong>",
		"Hello &lt;world&gt;",
		`data-video="vid1"`,
		"https://i.ytimg.com/vi/vid1/mqdefault.jpg",
		`src="data:image/png;base64,`,
		"Full description",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("bulletin missing %q", want)
		}
	}
	if strings.Contains(body, "ZgotmplZ") {
		t.Error("template sanitised a URL it should have trusted")
	}

	rec = get(t, srv.Handler(), "/api/bulletin?hours=48")
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"video_id":"vid1"`) {
		t.Errorf("api/bulletin: %d %s", rec.Code, rec.Body.String())
	}
}

func TestSetupPage(t *testing.T) {
	srv, svc := beeServer(t)
	h := srv.Handler()

	if rec := get(t, h, "/setup"); rec.Code != 200 || !strings.Contains(rec.Body.String(), "YouTube Data API key") {
		t.Fatalf("setup GET: %d", rec.Code)
	}

	// Saving without a working API just records the sync error.
	oldBase := dailybee.APIBase
	dailybee.APIBase = "http://127.0.0.1:1"
	defer func() { dailybee.APIBase = oldBase }()
	form := url.Values{"api_key": {"AIzaSECRET9876"}, "channel": {"UCalphaalphaalphaalpha00"}, "hours": {"48"}, "hide_shorts": {"on"}}
	req := httptest.NewRequest("POST", "/setup", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusSeeOther {
		t.Fatalf("setup POST: %d", rec.Code)
	}
	cfg := svc.StoredConfig()
	if cfg.APIKey != "AIzaSECRET9876" || cfg.Hours != 48 || !cfg.HideShorts || cfg.Prune {
		t.Errorf("config not saved: %+v", cfg)
	}

	rec = get(t, h, "/setup?saved=1")
	body := rec.Body.String()
	if !strings.Contains(body, "could not reach the YouTube API") {
		t.Error("sync error not shown")
	}
	if strings.Contains(body, "AIzaSECRET9876") {
		t.Error("setup page leaked the API key")
	}
	if !strings.Contains(body, "9876") || !strings.Contains(body, `class="notice err"`) {
		t.Error("setup page should show the masked key and the sync error")
	}
	if rec := get(t, h, "/api/settings"); strings.Contains(rec.Body.String(), "SECRET") {
		t.Error("/api/settings leaked the API key")
	}
}

func TestBulletinRequiresLogin(t *testing.T) {
	srv, _ := beeServer(t)
	srv.Auth = NewLocalAuthProvider("user", "pass", "")
	rec := get(t, srv.Handler(), "/")
	if rec.Code != http.StatusFound || rec.Header().Get("Location") != "/reader" {
		t.Errorf("expected redirect to /reader, got %d %q", rec.Code, rec.Header().Get("Location"))
	}
	if rec := get(t, srv.Handler(), "/api/bulletin"); rec.Code != http.StatusUnauthorized {
		t.Errorf("api/bulletin without login: %d", rec.Code)
	}
}
