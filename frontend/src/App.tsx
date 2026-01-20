import { PongGame } from "./components/PongGame";
import "./index.css";

export function App() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-[#0a0a0a] to-[#1a1a1a] flex items-center justify-center">
      <PongGame />
    </div>
  );
}

export default App;
