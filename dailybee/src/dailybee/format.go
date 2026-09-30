package dailybee

import (
	"fmt"
	"strconv"
	"time"
)

// FormatClock renders a video length the way YouTube does: 4:05, 1:02:03.
func FormatClock(d time.Duration) string {
	if d <= 0 {
		return ""
	}
	s := int(d.Round(time.Second).Seconds())
	h, m, sec := s/3600, (s%3600)/60, s%60
	if h > 0 {
		return fmt.Sprintf("%d:%02d:%02d", h, m, sec)
	}
	return fmt.Sprintf("%d:%02d", m, sec)
}

// FormatWatchTime renders a total like "3 h 20 min" or "45 min".
func FormatWatchTime(d time.Duration) string {
	if d <= 0 {
		return ""
	}
	m := int(d.Round(time.Minute).Minutes())
	if m < 1 {
		return "under a minute"
	}
	if m < 60 {
		return fmt.Sprintf("%d min", m)
	}
	if m%60 == 0 {
		return fmt.Sprintf("%d h", m/60)
	}
	return fmt.Sprintf("%d h %d min", m/60, m%60)
}

// FormatCount renders view counts compactly: 950, 12K, 1.2M.
func FormatCount(n int64) string {
	switch {
	case n <= 0:
		return ""
	case n < 1_000:
		return strconv.FormatInt(n, 10)
	case n < 10_000:
		return trimZero(fmt.Sprintf("%.1f", float64(n)/1_000)) + "K"
	case n < 1_000_000:
		return strconv.FormatInt(n/1_000, 10) + "K"
	case n < 10_000_000:
		return trimZero(fmt.Sprintf("%.1f", float64(n)/1_000_000)) + "M"
	case n < 1_000_000_000:
		return strconv.FormatInt(n/1_000_000, 10) + "M"
	default:
		return trimZero(fmt.Sprintf("%.1f", float64(n)/1_000_000_000)) + "B"
	}
}

func trimZero(s string) string {
	if len(s) > 2 && s[len(s)-2:] == ".0" {
		return s[:len(s)-2]
	}
	return s
}

// FormatWindow describes the bulletin window: "24 hours", "2 days", "7 days".
func FormatWindow(hours int) string {
	if hours%24 == 0 && hours > 24 {
		return fmt.Sprintf("%d days", hours/24)
	}
	if hours == 1 {
		return "hour"
	}
	return fmt.Sprintf("%d hours", hours)
}
