// Package dailybee turns the yarr feed reader into a daily YouTube bulletin.
//
// Quota-friendly design: the YouTube Data API (and the user's API key) is only
// used to discover which channels the user is subscribed to and, optionally,
// to look up video durations / view counts in batches of 50. New uploads are
// discovered through each channel's public RSS feed, which yarr already knows
// how to poll and which costs no API quota at all.
package dailybee

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// APIBase is the YouTube Data API v3 endpoint. Overridden in tests.
var APIBase = "https://www.googleapis.com/youtube/v3"

var httpClient = &http.Client{Timeout: 30 * time.Second}

// Channel is a YouTube channel the user is subscribed to.
type Channel struct {
	ID          string
	Title       string
	Description string
	Thumbnail   string
}

func (c Channel) FeedURL() string { return ChannelFeedURL(c.ID) }
func (c Channel) URL() string     { return "https://www.youtube.com/channel/" + c.ID }

// ChannelFeedURL returns the public Atom feed with a channel's latest uploads.
func ChannelFeedURL(channelID string) string {
	return "https://www.youtube.com/feeds/videos.xml?channel_id=" + url.QueryEscape(channelID)
}

// ChannelIDFromFeedURL extracts the channel id from a YouTube channel feed URL.
// It returns "" for anything that is not a YouTube channel feed.
func ChannelIDFromFeedURL(feedURL string) string {
	u, err := url.Parse(feedURL)
	if err != nil {
		return ""
	}
	host := strings.TrimPrefix(u.Hostname(), "www.")
	if host != "youtube.com" || u.Path != "/feeds/videos.xml" {
		return ""
	}
	return u.Query().Get("channel_id")
}

// VideoDetails is the optional per-video enrichment fetched from videos.list.
type VideoDetails struct {
	Duration time.Duration
	Views    int64
	// LiveBroadcastContent is "none", "live" or "upcoming".
	LiveBroadcastContent string
}

// APIError is an error response from the YouTube Data API.
type APIError struct {
	Status  int
	Reason  string
	Message string
}

func (e *APIError) Error() string {
	switch e.Reason {
	case "subscriptionForbidden":
		return "YouTube refused to list the subscriptions of this channel. " +
			"Make them public in YouTube → Settings → Privacy → \"Keep all my subscriptions private\" (turn it off)."
	case "keyInvalid", "badRequest":
		if strings.Contains(strings.ToLower(e.Message), "api key") {
			return "The YouTube API key is not valid. Check it in Google Cloud Console → APIs & Services → Credentials."
		}
	case "accessNotConfigured", "SERVICE_DISABLED":
		return "The YouTube Data API v3 is not enabled for this API key's Google Cloud project."
	case "quotaExceeded", "dailyLimitExceeded":
		return "The daily YouTube API quota for this key is used up. It resets at midnight Pacific Time."
	case "subscriberNotFound", "channelNotFound":
		return "YouTube could not find that channel."
	}
	if e.Message != "" {
		return fmt.Sprintf("YouTube API error (%d %s): %s", e.Status, e.Reason, e.Message)
	}
	return fmt.Sprintf("YouTube API error: HTTP %d", e.Status)
}

func apiGet(ctx context.Context, endpoint string, params url.Values, dst any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, APIBase+"/"+endpoint+"?"+params.Encode(), nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "application/json")
	res, err := httpClient.Do(req)
	if err != nil {
		// *url.Error embeds the request URL, which contains the API key.
		var urlErr *url.Error
		if errors.As(err, &urlErr) {
			err = urlErr.Err
		}
		return fmt.Errorf("could not reach the YouTube API: %w", err)
	}
	defer res.Body.Close()
	body, err := io.ReadAll(io.LimitReader(res.Body, 16<<20))
	if err != nil {
		return err
	}
	if res.StatusCode != http.StatusOK {
		var e struct {
			Error struct {
				Message string `json:"message"`
				Status  string `json:"status"`
				Errors  []struct {
					Reason string `json:"reason"`
				} `json:"errors"`
			} `json:"error"`
		}
		apiErr := &APIError{Status: res.StatusCode}
		if json.Unmarshal(body, &e) == nil {
			apiErr.Message = e.Error.Message
			apiErr.Reason = e.Error.Status
			if len(e.Error.Errors) > 0 && e.Error.Errors[0].Reason != "" {
				apiErr.Reason = e.Error.Errors[0].Reason
			}
		}
		return apiErr
	}
	return json.Unmarshal(body, dst)
}

var (
	channelIDRe = regexp.MustCompile(`^UC[0-9A-Za-z_-]{22}$`)
	handleRe    = regexp.MustCompile(`^@?[0-9A-Za-z._-]{3,100}$`)
)

// ResolveChannelID accepts a channel id (UC...), a handle (@name) or a
// channel URL (youtube.com/channel/UC..., youtube.com/@name) and returns the
// channel id. Resolving a handle costs 1 quota unit.
func ResolveChannelID(ctx context.Context, apiKey, input string) (string, error) {
	input = strings.TrimSpace(input)
	if input == "" {
		return "", errors.New("no YouTube channel given")
	}
	if u, err := url.Parse(input); err == nil && u.Host != "" {
		parts := strings.Split(strings.Trim(u.Path, "/"), "/")
		switch {
		case len(parts) >= 2 && parts[0] == "channel":
			input = parts[1]
		case len(parts) >= 1 && strings.HasPrefix(parts[0], "@"):
			input = parts[0]
		default:
			return "", fmt.Errorf("unrecognised YouTube channel URL %q", input)
		}
	}
	if channelIDRe.MatchString(input) {
		return input, nil
	}
	if !handleRe.MatchString(input) {
		return "", fmt.Errorf("%q is not a channel id, @handle or channel URL", input)
	}
	handle := "@" + strings.TrimPrefix(input, "@")
	var res struct {
		Items []struct {
			ID string `json:"id"`
		} `json:"items"`
	}
	params := url.Values{"part": {"id"}, "forHandle": {handle}, "key": {apiKey}}
	if err := apiGet(ctx, "channels", params, &res); err != nil {
		return "", err
	}
	if len(res.Items) == 0 {
		return "", fmt.Errorf("no YouTube channel found for %s", handle)
	}
	return res.Items[0].ID, nil
}

// ListSubscriptions returns every channel the given channel is subscribed to.
// With a plain API key (no OAuth) this only works when the channel's
// subscriptions are public. Costs 1 quota unit per 50 subscriptions.
func ListSubscriptions(ctx context.Context, apiKey, channelID string) ([]Channel, error) {
	var channels []Channel
	pageToken := ""
	for page := 0; page < 100; page++ {
		var res struct {
			NextPageToken string `json:"nextPageToken"`
			Items         []struct {
				Snippet struct {
					Title       string `json:"title"`
					Description string `json:"description"`
					ResourceID  struct {
						ChannelID string `json:"channelId"`
					} `json:"resourceId"`
					Thumbnails map[string]struct {
						URL string `json:"url"`
					} `json:"thumbnails"`
				} `json:"snippet"`
			} `json:"items"`
		}
		params := url.Values{
			"part":       {"snippet"},
			"channelId":  {channelID},
			"maxResults": {"50"},
			"order":      {"alphabetical"},
			"key":        {apiKey},
		}
		if pageToken != "" {
			params.Set("pageToken", pageToken)
		}
		if err := apiGet(ctx, "subscriptions", params, &res); err != nil {
			return nil, err
		}
		for _, it := range res.Items {
			if it.Snippet.ResourceID.ChannelID == "" {
				continue
			}
			thumb := it.Snippet.Thumbnails["default"].URL
			if thumb == "" {
				thumb = it.Snippet.Thumbnails["medium"].URL
			}
			channels = append(channels, Channel{
				ID:          it.Snippet.ResourceID.ChannelID,
				Title:       it.Snippet.Title,
				Description: it.Snippet.Description,
				Thumbnail:   thumb,
			})
		}
		if res.NextPageToken == "" {
			break
		}
		pageToken = res.NextPageToken
	}
	return channels, nil
}

// GetVideoDetails looks up duration, views and live status for the given
// videos, 50 per request (1 quota unit each).
func GetVideoDetails(ctx context.Context, apiKey string, ids []string) (map[string]VideoDetails, error) {
	result := make(map[string]VideoDetails, len(ids))
	for start := 0; start < len(ids); start += 50 {
		end := min(start+50, len(ids))
		var res struct {
			Items []struct {
				ID      string `json:"id"`
				Snippet struct {
					LiveBroadcastContent string `json:"liveBroadcastContent"`
				} `json:"snippet"`
				ContentDetails struct {
					Duration string `json:"duration"`
				} `json:"contentDetails"`
				Statistics struct {
					ViewCount string `json:"viewCount"`
				} `json:"statistics"`
			} `json:"items"`
		}
		params := url.Values{
			"part":       {"snippet,contentDetails,statistics"},
			"id":         {strings.Join(ids[start:end], ",")},
			"maxResults": {"50"},
			"key":        {apiKey},
		}
		if err := apiGet(ctx, "videos", params, &res); err != nil {
			return result, err
		}
		for _, it := range res.Items {
			views, _ := strconv.ParseInt(it.Statistics.ViewCount, 10, 64)
			result[it.ID] = VideoDetails{
				Duration:             ParseISODuration(it.ContentDetails.Duration),
				Views:                views,
				LiveBroadcastContent: it.Snippet.LiveBroadcastContent,
			}
		}
	}
	return result, nil
}

var isoDurationRe = regexp.MustCompile(`^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$`)

// ParseISODuration parses the ISO-8601 durations YouTube uses, e.g. "PT1H2M3S".
func ParseISODuration(s string) time.Duration {
	m := isoDurationRe.FindStringSubmatch(s)
	if m == nil {
		return 0
	}
	units := []time.Duration{24 * time.Hour, time.Hour, time.Minute, time.Second}
	var d time.Duration
	for i, unit := range units {
		if m[i+1] != "" {
			n, _ := strconv.Atoi(m[i+1])
			d += time.Duration(n) * unit
		}
	}
	return d
}

// fetchImage downloads a small image (a channel avatar) for use as a feed icon.
func fetchImage(ctx context.Context, src string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, src, nil)
	if err != nil {
		return nil, err
	}
	res, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d", res.StatusCode)
	}
	return io.ReadAll(io.LimitReader(res.Body, 1<<20))
}
