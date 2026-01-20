import { serve } from "bun";
import index from "./index.html";

const server = serve({
  routes: {
    // Serve index.html for all unmatched routes.
    "/*": index,

    "/api/hello": {
      async GET(req) {
        return Response.json({
          message: "Hello, world!",
          method: "GET",
        });
      },
      async PUT(req) {
        return Response.json({
          message: "Hello, world!",
          method: "PUT",
        });
      },
    },

    "/api/hello/:name": async req => {
      const name = req.params.name;
      return Response.json({
        message: `Hello, ${name}!`,
      });
    },

    "/api/game-data": {
      async POST(req) {
        try {
          const body = await req.json();
          const { frames, date } = body;

          if (!frames || !Array.isArray(frames)) {
            return Response.json(
              { error: "Invalid request: frames array is required" },
              { status: 400 }
            );
          }

          // TODO: Connect to PostgreSQL and insert frame data
          // This is a placeholder - you'll implement the actual DB logic in the backend
          console.log(`Received ${frames.length} frames from game on ${date}`);

          // For now, just acknowledge receipt
          return Response.json({
            success: true,
            message: `Received ${frames.length} frames`,
            date: date,
          });
        } catch (error) {
          console.error("Error processing game data:", error);
          return Response.json(
            { error: "Failed to process game data" },
            { status: 500 }
          );
        }
      },
    },
  },

  development: process.env.NODE_ENV !== "production" && {
    // Enable browser hot reloading in development
    hmr: true,

    // Echo console logs from the browser to the server
    console: true,
  },
});

console.log(`🚀 Server running at ${server.url}`);
