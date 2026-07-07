import "./OsrsLoadingBar.css";

interface OsrsLoadingBarProps {
    text: string;
    progress: number;
}

export function OsrsLoadingBar({ text, progress }: OsrsLoadingBarProps): JSX.Element {
    const safeProgress = Math.max(0, Math.min(progress, 100));
    return (
        <div className="loading-bar">
            <div className="loading-bar-progress-container">
                <div className="loading-bar-progress" style={{ width: safeProgress + "%" }}></div>
            </div>
            <div className="loading-bar-text">
                {text} - {safeProgress}%
            </div>
        </div>
    );
}
