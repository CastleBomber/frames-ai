"use client";

import { useMemo, useState, type ChangeEvent } from "react";

type Move = {
  name: string;
  pose: string;
};

const moves: Move[] = [
  { name: "Groove", pose: "groove" },
  { name: "Shuffle", pose: "shuffle" },
  { name: "Spin", pose: "spin" },
  { name: "Bounce", pose: "bounce" },
  { name: "Freestyle", pose: "freestyle" },
];

function WingMark() {
  return (
    <span className="wing-mark" aria-hidden="true">
      <i /><i /><i /><i />
    </span>
  );
}

function Sparkle({ small = false }: { small?: boolean }) {
  return <span className={small ? "sparkle sparkle--small" : "sparkle"} aria-hidden="true">✦</span>;
}

function Dancer({
  pose = "hero",
  colorful = false,
}: {
  pose?: string;
  colorful?: boolean;
}) {
  return (
    <div className={`dancer dancer--${pose}${colorful ? " dancer--colorful" : ""}`} aria-hidden="true">
      <span className="dancer__glow" />
      <span className="dancer__head" />
      <span className="dancer__neck" />
      <span className="dancer__body" />
      <span className="dancer__arm dancer__arm--left"><i /></span>
      <span className="dancer__arm dancer__arm--right"><i /></span>
      <span className="dancer__leg dancer__leg--left"><i /></span>
      <span className="dancer__leg dancer__leg--right"><i /></span>
    </div>
  );
}

function Waveform() {
  const bars = [18, 38, 62, 32, 72, 51, 84, 44, 67, 35, 78, 55, 91, 48, 69, 37, 82, 58, 73, 30, 64, 42, 76, 52, 88, 39, 71, 47, 80, 33, 59, 26];
  return (
    <span className="waveform" aria-hidden="true">
      {bars.map((height, index) => <i key={index} style={{ height: `${height}%` }} />)}
    </span>
  );
}

function StageLights() {
  return (
    <div className="stage-lights" aria-hidden="true">
      <span className="beam beam--one" /><span className="beam beam--two" />
      <span className="beam beam--three" /><span className="beam beam--four" />
      <span className="tower tower--left"><i /><i /><i /><i /></span>
      <span className="tower tower--right"><i /><i /><i /><i /></span>
      <span className="equalizer equalizer--left">{Array.from({ length: 18 }, (_, index) => <i key={index} />)}</span>
      <span className="equalizer equalizer--right">{Array.from({ length: 18 }, (_, index) => <i key={index} />)}</span>
    </div>
  );
}

function fileName(event: ChangeEvent<HTMLInputElement>, fallback: string) {
  return event.target.files?.[0]?.name ?? fallback;
}

export default function Home() {
  const [selectedMoves, setSelectedMoves] = useState(() => new Set(["Groove", "Spin", "Bounce"]));
  const [previewing, setPreviewing] = useState(false);
  const [song, setSong] = useState("Funky Nights");
  const [drawing, setDrawing] = useState("hero-drawing.png");
  const [created, setCreated] = useState(false);

  const selectedCount = selectedMoves.size;
  const moveSummary = useMemo(() => [...selectedMoves].join(", "), [selectedMoves]);

  function toggleMove(name: string) {
    setSelectedMoves((current) => {
      const next = new Set(current);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
    setCreated(false);
  }

  function createDance() {
    setCreated(true);
    setPreviewing(true);
    window.setTimeout(() => setCreated(false), 3200);
  }

  return (
    <main className="site-frame">
      <div className="ambient ambient--mint" aria-hidden="true" />
      <div className="ambient ambient--violet" aria-hidden="true" />

      <section className="app-shell" aria-label="Angels AI dance creator">
        <nav className="topbar">
          <a className="brand" href="#stage" aria-label="Angels AI home">
            <WingMark />
            <span>Angels <strong>AI</strong></span>
          </a>
          <div className="topbar__actions">
            <button className="heroes-button" type="button"><span aria-hidden="true">♙</span> My Heroes</button>
            <button className="gold-button gold-button--small" type="button" onClick={() => document.querySelector("#creator")?.scrollIntoView({ behavior: "smooth" })}>
              <Sparkle small /> Create
            </button>
          </div>
        </nav>

        <section className={`stage${previewing ? " is-previewing" : ""}`} id="stage">
          <StageLights />
          <div className="skills-badge"><span>★</span> {selectedCount} skill{selectedCount === 1 ? "" : "s"} equipped</div>
          <div className="hero-dancer"><Dancer pose={previewing ? "groove" : "hero"} /></div>
          <div className="stage-platform"><i /><b /></div>
          <button className="preview-button" type="button" onClick={() => setPreviewing((value) => !value)} aria-pressed={previewing}>
            <span>{previewing ? "Ⅱ" : "▶"}</span>{previewing ? "Pause preview" : "Preview dance"}
          </button>
        </section>

        <section className="moves-section" aria-labelledby="moves-heading">
          <header className="section-title">
            <span className="section-icon" aria-hidden="true">⌁</span>
            <div><h1 id="moves-heading">Choose your hero&apos;s moves</h1><p>{selectedCount ? moveSummary : "Pick at least one move"}</p></div>
          </header>

          <div className="moves-rail">
            <button className="rail-arrow" type="button" aria-label="Previous moves">‹</button>
            <div className="move-grid">
              {moves.map((move) => {
                const selected = selectedMoves.has(move.name);
                return (
                  <button
                    className={`move-card${selected ? " is-selected" : ""}`}
                    type="button"
                    key={move.name}
                    onClick={() => toggleMove(move.name)}
                    aria-pressed={selected}
                  >
                    <span className="move-card__check">{selected ? "✓" : "+"}</span>
                    <span className="move-card__figure"><Dancer pose={move.pose} /></span>
                    <strong>{move.name}</strong>
                  </button>
                );
              })}
            </div>
            <button className="rail-arrow" type="button" aria-label="Next moves">›</button>
          </div>
        </section>

        <section className="creator" id="creator" aria-label="Three-step dance creator">
          <article className="creator-step song-step">
            <header><span className="step-number">1</span><h2>Add a song</h2></header>
            <div className="track-card">
              <span className="music-tile" aria-hidden="true">♫</span>
              <span className="track-copy"><strong>{song}</strong><small>128 BPM · Funk<br />02:48</small></span>
              <label className="change-button">Change<input type="file" accept="audio/*" onChange={(event) => setSong(fileName(event, song))} /></label>
              <button className="round-play" type="button" aria-label="Play song">▶</button>
              <Waveform />
              <time>2:48</time>
            </div>
            <label className="upload-zone">
              <span className="upload-symbol">↥</span>
              <span><strong>Upload your song</strong><small>MP3, WAV, M4A up to 50MB</small></span>
              <input type="file" accept="audio/*" onChange={(event) => setSong(fileName(event, song))} />
            </label>
          </article>

          <span className="flow-arrow" aria-hidden="true">⟶</span>

          <article className="creator-step character-step">
            <header><span className="step-number">2</span><h2>Add your character</h2></header>
            <label className="drawing-card">
              <span className="drawing-paper"><span>☆</span><Dancer pose="groove" colorful /><i>♡</i></span>
              <span className="drawing-upload"><span>↥</span> {drawing}</span>
              <input type="file" accept="image/png,image/jpeg" onChange={(event) => setDrawing(fileName(event, drawing))} />
            </label>
            <div className="transformation-arrow" aria-hidden="true"><span>→</span><Sparkle small /></div>
            <div className="character-preview"><Dancer pose="groove" colorful /><Sparkle /><Sparkle small /></div>
          </article>

          <span className="flow-arrow" aria-hidden="true">⟶</span>

          <article className="creator-step result-step">
            <header><span className="step-number">3</span><h2>Dance on beat</h2></header>
            <div className={`result-stage${previewing ? " is-previewing" : ""}`}>
              <StageLights />
              <Dancer pose="groove" colorful />
              <span className="result-floor" />
            </div>
            <button className="gold-button create-dance-button" type="button" onClick={createDance}>
              <Sparkle /> {created ? "Your hero is alive!" : "Bring my hero to life"}
            </button>
          </article>
        </section>
      </section>

      <footer className="site-footer">
        <span><i /> Mint Halo</span><span><i /> Celestial Violet</span><span><i /> Golden Beat</span>
        <p>Roadmap V4 · Interactive website shell</p>
      </footer>
    </main>
  );
}
