package dailybee

import (
	"strings"
	"testing"

	"github.com/nkanaev/yarr/src/parser"
	"github.com/nkanaev/yarr/src/storage/model"
	"github.com/nkanaev/yarr/src/worker"
)

// Trimmed copy of what https://www.youtube.com/feeds/videos.xml?channel_id=... returns.
const youtubeFeed = `<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns:yt="http://www.youtube.com/xml/schemas/2015" xmlns:media="http://search.yahoo.com/mrss/" xmlns="http://www.w3.org/2005/Atom">
 <link rel="self" href="http://www.youtube.com/feeds/videos.xml?channel_id=UCalphaalphaalphaalpha00"/>
 <id>yt:channel:alphaalphaalphaalpha00</id>
 <yt:channelId>alphaalphaalphaalpha00</yt:channelId>
 <title>Alpha Channel</title>
 <link rel="alternate" href="https://www.youtube.com/channel/UCalphaalphaalphaalpha00"/>
 <published>2015-01-01T00:00:00+00:00</published>
 <entry>
  <id>yt:video:dQw4w9WgXcQ</id>
  <yt:videoId>dQw4w9WgXcQ</yt:videoId>
  <yt:channelId>UCalphaalphaalphaalpha00</yt:channelId>
  <title>A brand new video &amp; more</title>
  <link rel="alternate" href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"/>
  <author><name>Alpha Channel</name><uri>https://www.youtube.com/channel/UCalphaalphaalphaalpha00</uri></author>
  <published>2026-09-29T15:00:06+00:00</published>
  <updated>2026-09-29T16:12:40+00:00</updated>
  <media:group>
   <media:title>A brand new video &amp; more</media:title>
   <media:content url="https://www.youtube.com/v/dQw4w9WgXcQ?version=3" type="application/x-shockwave-flash" width="640" height="390"/>
   <media:thumbnail url="https://i4.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg" width="480" height="360"/>
   <media:description>In today's video we look at bees.
Sponsored by honey. https://example.com/honey

Chapters:
0:00 Intro</media:description>
   <media:community>
    <media:starRating count="1200" average="5.00" min="1" max="5"/>
    <media:statistics views="45678"/>
   </media:community>
  </media:group>
 </entry>
</feed>`

func TestYouTubeFeedParsing(t *testing.T) {
	feed, err := parser.Parse(strings.NewReader(youtubeFeed))
	if err != nil {
		t.Fatal(err)
	}
	items := worker.ConvertItems(feed.Items, model.Feed{Id: 1})
	if len(items) != 1 {
		t.Fatalf("got %d items", len(items))
	}
	v := videoFromItem(items[0])
	if v.VideoID != "dQw4w9WgXcQ" {
		t.Errorf("video id = %q (guid %q)", v.VideoID, items[0].GUID)
	}
	if v.Title != "A brand new video & more" || v.URL != "https://www.youtube.com/watch?v=dQw4w9WgXcQ" {
		t.Errorf("title/url = %q / %q", v.Title, v.URL)
	}
	if v.Published.IsZero() || v.Published.Year() != 2026 {
		t.Errorf("published = %v", v.Published)
	}
	if v.Summary != "In today's video we look at bees. Sponsored by honey. https://example.com/honey" {
		t.Errorf("summary = %q", v.Summary)
	}
	if !strings.Contains(v.Description, "0:00 Intro") {
		t.Errorf("description = %q", v.Description)
	}
}
