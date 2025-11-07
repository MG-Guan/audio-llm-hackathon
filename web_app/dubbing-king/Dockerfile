# ---------- Build Stage ----------
FROM node:22-alpine AS build

# Set working directory
WORKDIR /app

# Enable pnpm via Corepack (built into Node 22)
RUN corepack enable && corepack prepare pnpm@latest --activate

# Copy dependency files first (for layer caching)
COPY package.json pnpm-lock.yaml ./

# Install dependencies (use frozen lockfile for reproducible builds)
RUN pnpm install --frozen-lockfile

# Copy the rest of the Angular project
COPY . .

# Build the Angular app (adjust configuration if needed)
RUN pnpm run build

# ---------- Serve Stage ----------
FROM node:22-alpine AS serve

WORKDIR /app

# Install sirv-cli globally (lightweight SPA server)
RUN npm install -g sirv-cli

# Copy built Angular artifacts from the previous stage
COPY --from=build /app/dist/dubbing-king/browser ./dist

# Expose port 4200
EXPOSE 4200

# Start the web server
CMD ["sirv", "dist", "--single", "--host", "0.0.0.0", "--port", "4200"]
