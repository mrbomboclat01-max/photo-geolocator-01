// photo-geolocator-01 entry
//
// Strategy: two-stage per photo.
//   Stage 1 (image call): structured chain-of-thought analysis extracting visual clues,
//     culminating in a region estimate + initial coordinates.  max_tokens=400.
//   Stage 2 (text-only call): takes stage-1 analysis as context, reasons once more to
//     produce final precise coordinates.  max_tokens=200.  Very cheap (no image re-send).
//
// Stage-2 calls are text-only, so their cost is negligible; the budget is dominated by
// the 25 image calls, same as the baseline.  Both stages run at CONCURRENCY=6.
//
// Determinism: every request is a pure function of the photo bytes and the fixed prompts.
// Calls run in parallel; answers are posted in photo order after all workers finish.

const MODEL = 'claude-sonnet-5-5';
const CONCURRENCY = 6;
const MAX_TOKENS_STAGE1 = 400;
const MAX_TOKENS_STAGE2 = 200;
const RETRY_STATUSES = [408, 429, 500, 502, 503, 504, 529];
const RETRIES = 3;

const SYSTEM = `You are a world-class geolocation expert. You have deep knowledge of:
- Road markings, guardrail styles, traffic signs and infrastructure by country
- Vegetation, biomes and plant species worldwide
- Architectural styles, building materials and construction eras by region
- Vehicle makes/models and license plate designs by country
- Written scripts, languages and their geographic distribution
- Terrain, topology and geographic features worldwide
You reliably identify where street-level photos were taken, even from subtle clues.`;

const STAGE1_PROMPT = `Analyze this street-level photo to identify where it was taken.

Work through each category:
LANGUAGE/TEXT: Any visible script, words, place names, brand names, signs?
VEGETATION: Plant species, biome clue (tropical, temperate, Mediterranean, arid, boreal)?
ROAD/INFRA: Road quality, markings, guardrail style, traffic sign design, sidewalk type?
VEHICLES: Visible makes/models, traffic side (LHD or RHD), license plate colour/style?
ARCHITECTURE: Building materials, style, era, density (urban/rural/suburban)?
TERRAIN: Mountains, coast, plains, valley? Sky/climate indicators?
UTILITY: Power line style, pole type, electrical infrastructure?

Then reason: given all clues → most likely COUNTRY → most likely REGION within that country → precise coordinates.

End with exactly one line of JSON (nothing else after it):
{"lat": <decimal degrees -90 to 90>, "lon": <decimal degrees -180 to 180>}`;

const STAGE2_PROMPT_PREFIX = `A street-level photo was analyzed and produced this observation:

`;

const STAGE2_PROMPT_SUFFIX = `

Using that analysis, reason carefully about the most precise GPS location possible.
Consider the specific clues mentioned to narrow down the exact region, city or road.

Reply with exactly one line of JSON and nothing else:
{"lat": <decimal degrees -90 to 90>, "lon": <decimal degrees -180 to 180>}`;

interface Photo {
  id: string;
  url: string;
  answered: boolean;
}

interface Guess {
  id: string;
  lat: number;
  lon: number;
}

function need(name: string): string {
  const v = process.env[name];
  if (v === undefined || v === '') throw new Error(`${name} is not set`);
  return v;
}

const envUrl = need('MAKEFUN_ENV_URL').replace(/\/+$/, '');
const anthropicUrl = need('ANTHROPIC_BASE_URL').replace(/\/+$/, '');
const apiKey = need('ANTHROPIC_API_KEY');

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

async function callAnthropic(body: object): Promise<string | null> {
  for (let attempt = 0; ; attempt += 1) {
    const res = await fetch(`${anthropicUrl}/v1/messages`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify(body),
    });
    if (res.ok) {
      const msg = (await res.json()) as { content?: { type: string; text?: string }[] };
      return (msg.content ?? [])
        .filter((b) => b.type === 'text')
        .map((b) => b.text ?? '')
        .join('\n');
    }
    const text = await res.text();
    if (
      RETRY_STATUSES.includes(res.status) &&
      res.headers.get('x-should-retry') !== 'false' &&
      attempt < RETRIES
    ) {
      await sleep(2000 * (attempt + 1));
      continue;
    }
    process.stderr.write(`model call refused (HTTP ${res.status}): ${text.slice(0, 300)}\n`);
    return null;
  }
}

const isLat = (v: unknown): v is number =>
  typeof v === 'number' && Number.isFinite(v) && v >= -90 && v <= 90;
const isLon = (v: unknown): v is number =>
  typeof v === 'number' && Number.isFinite(v) && v >= -180 && v <= 180;

function parseCoordinates(text: string): { lat: number; lon: number } | null {
  // Try every JSON object in the text for lat/lon
  for (const m of text.matchAll(/\{[^{}]*\}/g)) {
    try {
      const o = JSON.parse(m[0]) as Record<string, unknown>;
      const lat = o.lat ?? o.latitude;
      const lon = o.lon ?? o.lng ?? o.longitude;
      if (isLat(lat) && isLon(lon)) return { lat: lat as number, lon: lon as number };
    } catch {
      // not JSON, try next
    }
  }
  // Fallback: first two numbers that look like lat/lon
  const nums = [...text.matchAll(/-?\d+(?:\.\d+)?/g)].map((m) => Number(m[0]));
  for (let i = 0; i + 1 < nums.length; i += 1) {
    if (isLat(nums[i]) && isLon(nums[i + 1])) return { lat: nums[i]!, lon: nums[i + 1]! };
  }
  return null;
}

async function locatePhoto(photo: Photo): Promise<Guess | null> {
  // Fetch image bytes
  const jpeg = Buffer.from(await (await fetch(`${envUrl}${photo.url}`)).arrayBuffer());

  // Stage 1: image + structured analysis prompt
  const stage1Text = await callAnthropic({
    model: MODEL,
    system: SYSTEM,
    max_tokens: MAX_TOKENS_STAGE1,
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'image',
            source: { type: 'base64', media_type: 'image/jpeg', data: jpeg.toString('base64') },
          },
          { type: 'text', text: STAGE1_PROMPT },
        ],
      },
    ],
  });

  if (stage1Text === null) return null;

  // Stage 2: text-only refinement using stage-1 analysis as context
  const stage2Text = await callAnthropic({
    model: MODEL,
    system: SYSTEM,
    max_tokens: MAX_TOKENS_STAGE2,
    messages: [
      {
        role: 'user',
        content: STAGE2_PROMPT_PREFIX + stage1Text + STAGE2_PROMPT_SUFFIX,
      },
    ],
  });

  // Prefer stage-2 coordinates; fall back to stage-1 if stage-2 failed or unparseable
  const coords =
    (stage2Text !== null ? parseCoordinates(stage2Text) : null) ??
    parseCoordinates(stage1Text);

  if (coords === null) {
    process.stderr.write(
      `no coordinates for ${photo.id}. s1: ${stage1Text.slice(0, 200)}  s2: ${(stage2Text ?? '').slice(0, 200)}\n`,
    );
    return null;
  }

  return { id: photo.id, lat: coords.lat, lon: coords.lon };
}

async function main(): Promise<number> {
  const listed = (await (await fetch(`${envUrl}/photos`)).json()) as { photos: Photo[] };
  const photos = listed.photos.filter((p) => !p.answered);

  const guesses: (Guess | null)[] = new Array(photos.length).fill(null);
  let next = 0;

  const worker = async (): Promise<void> => {
    while (next < photos.length) {
      const i = next;
      next += 1;
      guesses[i] = await locatePhoto(photos[i]!);
    }
  };

  await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));

  const answers = guesses.filter((g): g is Guess => g !== null);
  if (answers.length === 0) return 0;

  const res = await fetch(`${envUrl}/answers`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ answers }),
  });
  process.stdout.write(
    `${answers.length}/${photos.length} answered: HTTP ${res.status} ${await res.text()}\n`,
  );
  return res.ok ? 0 : 1;
}

try {
  process.exitCode = await main();
} catch (err) {
  process.stderr.write(`fatal: ${(err as Error).stack ?? String(err)}\n`);
  process.exitCode = 1;
}
