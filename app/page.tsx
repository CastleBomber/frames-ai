const moves = [
  { name: "Groove", pose: "groove", hue: "violet", selected: true },
  { name: "Shuffle", pose: "shuffle", hue: "cyan", selected: false },
  { name: "Spin", pose: "spin", hue: "pink", selected: true },
  { name: "Bounce", pose: "bounce", hue: "orange", selected: true },
  { name: "Freestyle", pose: "freestyle", hue: "lime", selected: false },
];

function WingMark() {
  return (
    <span className="wing-mark" aria-hidden="true">
      <span />
      <i />
      <b />
    </span>
  );
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
  const bars = [18, 35, 54, 28, 62, 45, 76, 32, 56, 82, 44, 69, 28, 52, 74, 38, 61, 30, 48, 72, 41, 58, 24, 46, 66, 33, 51, 21];
  return (
    <div className="waveform" aria-hidden="true">
      {bars.map((height, index) => (
        <span key={index} style={{ height: `${height}%` }} />
      ))}
    </div>
  );
}

function UploadIcon({ kind }: { kind: "music" | "image" }) {
  return (
    <span className={`upload-icon upload-icon--${kind}`} aria-hidden="true">
      {kind === "music" ? "♫" : "✦"}
    </span>
  );
}

export default function Home() {
  return (
    <main>
      <nav className="nav shell">
        <a className="brand" href="#top" aria-label="Angels AI home">
          <WingMark />
          <span>Angels <strong>AI</strong></span>
        </a>
        <div className="nav__links">
          <a href="#moves">Dance moves</a>
          <a href="#create">Create</a>
          <span className="nav__divider" />
          <button className="ghost-button" type="button">My heroes</button>
          <button className="avatar" type="button" aria-label="Account">A</button>
        </div>
      </nav>

      <section className="hero shell" id="top">
        <div className="eyebrow"><i /> Build your dancer</div>
        <h1>Teach your hero<br /><em>how to move.</em></h1>
        <p className="hero__copy">
          Choose their moves. Add your art. Pick a song.<br />
          Angels AI makes your character dance on beat.
        </p>

        <div className="stage">
          <span className="stage__aura stage__aura--one" />
          <span className="stage__aura stage__aura--two" />
          <span className="stage__grid" />
          <div className="stage__label"><span /> Your hero</div>
          <Dancer />
          <div className="stage__platform"><i /></div>
          <button type="button" className="play-button">
            <span>▶</span>
            Preview dance
          </button>
        </div>
      </section>

      <section className="moves shell" id="moves">
        <div className="section-heading">
          <div>
            <span className="step-label">01 · Choose their skills</span>
            <h2>Which moves will your hero know?</h2>
          </div>
          <p>Select as many as you like. We&apos;ll blend them to your beat.</p>
        </div>

        <div className="move-grid">
          {moves.map((move) => (
            <article
              className={`move-card move-card--${move.hue}${move.selected ? " is-selected" : ""}`}
              key={move.name}
            >
              <button type="button" className="select-control" aria-label={`${move.selected ? "Remove" : "Select"} ${move.name}`}>
                {move.selected ? "✓" : "+"}
              </button>
              <div className="move-card__visual"><Dancer pose={move.pose} colorful /></div>
              <div className="move-card__meta">
                <span>{move.name}</span>
                <small>{move.selected ? "Selected" : "Add move"}</small>
              </div>
            </article>
          ))}
        </div>

        <div className="carousel-controls">
          <span className="carousel-arrow">←</span>
          <i /><i className="active" /><i />
          <span className="carousel-arrow">→</span>
        </div>
      </section>

      <section className="create" id="create">
        <div className="shell">
          <div className="section-heading section-heading--create">
            <div>
              <span className="step-label">02 · Bring it to life</span>
              <h2>Your art. Your music. Their moment.</h2>
            </div>
            <span className="coming-soon">Prototype workflow</span>
          </div>

          <div className="workflow">
            <article className="workflow-card">
              <header>
                <span className="workflow-number">1</span>
                <div><h3>Add a song</h3><p>Choose a track for your hero.</p></div>
              </header>
              <div className="upload-zone">
                <UploadIcon kind="music" />
                <strong>Drop your song here</strong>
                <span>or browse audio files</span>
                <small>MP3, WAV · up to 20 MB</small>
              </div>
              <div className="mini-player">
                <button type="button" aria-label="Play">▶</button>
                <Waveform />
                <time>0:24</time>
              </div>
            </article>

            <div className="workflow-link" aria-hidden="true"><span>→</span></div>

            <article className="workflow-card">
              <header>
                <span className="workflow-number">2</span>
                <div><h3>Add your character</h3><p>Upload drawings from a few angles.</p></div>
              </header>
              <div className="upload-zone upload-zone--images">
                <UploadIcon kind="image" />
                <strong>Drop your drawings here</strong>
                <span>Front, side &amp; back work best</span>
                <div className="angle-row">
                  <i>+</i><i>+</i><i>+</i>
                </div>
              </div>
              <p className="privacy-note"><span>◇</span> Your artwork stays yours.</p>
            </article>

            <div className="workflow-link" aria-hidden="true"><span>→</span></div>

            <article className="workflow-card workflow-card--final">
              <header>
                <span className="workflow-number">3</span>
                <div><h3>Dance on beat</h3><p>We sync their skills to your song.</p></div>
              </header>
              <div className="result-preview">
                <span className="result-preview__light" />
                <Dancer pose="groove" colorful />
                <div className="result-preview__floor" />
                <div className="beat-pill"><i /> Beat synced</div>
              </div>
              <button type="button" className="create-button"><span>✦</span> Create my dance</button>
            </article>
          </div>
        </div>
      </section>

      <footer className="shell footer">
        <a className="brand brand--small" href="#top"><WingMark /><span>Angels <strong>AI</strong></span></a>
        <p>Turn your drawings into dancers.</p>
        <span>Roadmap V4 · Website shell</span>
      </footer>
    </main>
  );
}
