import { useEffect, useState } from "react";
import { VideoOff } from "lucide-react";

/**
 * Refreshing camera still. Preloads each new frame off-screen and swaps only once it has
 * decoded, so tiles never flash blank between frames. (Swapped for WebRTC via go2rtc later.)
 */
export function LiveImage({
  src,
  interval = 2000,
  className = "",
  alt = "",
  light = false,
  style,
}: {
  src: string;
  interval?: number;
  className?: string;
  alt?: string;
  /** Placeholder style for light surfaces (thumbnails in cards) vs. the dark video wall. */
  light?: boolean;
  style?: React.CSSProperties;
}) {
  const [shown, setShown] = useState<string | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const load = () => {
      const url = `${src}${src.includes("?") ? "&" : "?"}t=${Date.now()}`;
      const img = new Image();
      img.onload = () => {
        if (!alive) return;
        setShown(url);
        setOffline(false);
        timer = setTimeout(load, interval);
      };
      img.onerror = () => {
        if (!alive) return;
        setOffline(true);
        timer = setTimeout(load, interval * 3);
      };
      img.src = url;
    };
    load();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [src, interval]);

  if (shown) return <img src={shown} alt={alt} className={`object-cover ${className}`} style={style} draggable={false} />;
  if (offline)
    return (
      <div
        className={`flex items-center justify-center ${light ? "bg-sunken text-faint" : "bg-video text-white/40"} ${className}`}
        style={style}
      >
        <VideoOff size={16} strokeWidth={1.5} />
      </div>
    );
  return <div className={`${light ? "skeleton-light" : "skeleton"} ${className}`} style={style} />;
}
