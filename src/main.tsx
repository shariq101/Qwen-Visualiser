import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { useAtlas } from "./lib/data";
import "./styles.css";

function Root() {
  const { atlas, error, loaded, total } = useAtlas();

  if (error) {
    return (
      <div className="loading">
        <div className="t">Depth Atlas</div>
        <div className="err">
          Could not load the measured data ({error}).
          <br />
          Run <code>python experiments/suite.py all</code> to regenerate it into
          <code> public/data/</code>.
        </div>
      </div>
    );
  }
  if (!atlas) {
    return (
      <div className="loading">
        <div className="t">Depth Atlas</div>
        <div className="p">
          loading measured activations… {loaded}/{total}
        </div>
      </div>
    );
  }
  return <App atlas={atlas} />;
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
);
