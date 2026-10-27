-- v7 (freeze candidate): pressure is expressed as MISSING ink, not translucent ink.
-- Word band: solid 255 alpha, zero speckle. Frame rules: solid ink + starvation
-- clusters concentrated in the far quadrant + corner pool at the strike corner +
-- one drag run on the far rule. Per-stamp budgets keep the two objects distinct.
local OUT = "C:/Code Main/mk-sh-v/tmp/amber-art/"

local INK_GREEN = Color{ r=47,  g=107, b=71 }
local INK_RED   = Color{ r=158, g=59,  b=50 }
local PAPER     = Color{ r=239, g=231, b=210 }
local CLEAR     = Color{ r=0, g=0, b=0, a=0 }

local GLYPH = {
  A = { ".##.", "#..#", "#..#", "####", "#..#", "#..#" },
  P = { ".##.", "#..#", "#..#", ".##.", "#...", "#..." },
  R = { ".##.", "#..#", "#..#", ".##.", "#.#.", "#..#" },
  O = { ".##.", "#..#", "#..#", "#..#", "#..#", ".##." },
  V = { "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.." },
  E = { "####", "#...", "#...", "###.", "#...", "####" },
  D = { ".###", "#..#", "#..#", "#..#", "#..#", ".###" },
  N = { "#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#" },
  Y = { "#...#", "#...#", "#...#", ".#.#.", "..#..", "..#.." },
}

local function px(img, x, y, c) img:putPixel(x, y, c) end

local function draw_text(img, word, x0, y0, s, c)
  local x = x0
  for i = 1, #word do
    local gl = GLYPH[word:sub(i, i)]
    for row = 1, 6 do
      for col = 1, #gl[row] do
        if gl[row]:sub(col, col) == "#" then
          for dy = 0, s - 1 do
            for dx = 0, s - 1 do
              px(img, x + (col - 1) * s + dx, y0 + (row - 1) * s + dy, c)
            end
          end
        end
      end
    end
    x = x + #gl[1] * s + 2
  end
end

local function text_width(word, s)
  local w = 0
  for i = 1, #word do w = w + #GLYPH[word:sub(i, i)][1] * s end
  return w + (#word - 1) * 2
end

local function ring_rect(img, x0, y0, x1, y1, c)
  for x = x0, x1 do px(img, x, y0, c); px(img, x, y1, c) end
  for y = y0, y1 do px(img, x0, y, c); px(img, x1, y, c) end
end

local function clear_if_ink(img, x, y)
  if x >= 0 and x <= 79 and y >= 0 and y <= 47 and img:getPixel(x, y) ~= 0 then
    px(img, x, y, CLEAR)
  end
end

-- cfg: seed, pcx/pcy strike center, starve_big (large starvation patches, far side),
-- starve_sm, drag (pixels in the drag run)
local function make_stamp(word, ink, cfg)
  math.randomseed(cfg.seed)
  local W, H = 80, 48
  local spr = Sprite(W, H)
  local img = spr.cels[1].image
  ring_rect(img, 2, 2, W - 3, H - 3, ink)
  ring_rect(img, 6, 6, W - 7, H - 7, ink)
  local tw = text_width(word, 2)
  local x0 = math.floor((W - tw) / 2)
  local y0 = 18
  draw_text(img, word, x0, y0, 2, ink)   -- solid, untouched
  local function in_safe(x, y)
    return x >= x0 - 3 and x <= x0 + tw + 2 and y >= y0 - 4 and y <= y0 + 13
  end
  -- far side = the rule region farthest from the strike center
  local farx = (cfg.pcx < W / 2) and (W - 10) or 10
  local fary = (cfg.pcy < H / 2) and (H - 10) or 10
  -- large starvation patches: coherent missing-ink clusters near the far corner rules
  for _ = 1, cfg.starve_big do
    local cx = farx + math.random(-6, 6)
    local cy = fary + math.random(-6, 6)
    for _ = 1, math.random(4, 7) do
      local dx, dy = math.random(-1, 1), math.random(-1, 1)
      if not in_safe(cx + dx, cy + dy) then clear_if_ink(img, cx + dx, cy + dy) end
    end
  end
  -- small dropout clusters anywhere on the rules
  for _ = 1, cfg.starve_sm do
    local cx = math.random(4, W - 5)
    local cy = math.random(4, H - 5)
    local only_ring = (cy <= 8 or cy >= H - 9 or cx <= 8 or cx >= W - 9)
    if only_ring then
      for _ = 1, math.random(2, 3) do
        local dx, dy = math.random(-1, 1), math.random(-1, 1)
        if not in_safe(cx + dx, cy + dy) then clear_if_ink(img, cx + dx, cy + dy) end
      end
    end
  end
  -- drag run: a horizontal run of missing ink on the rule nearest the far side
  local dy0 = (fary > H / 2) and (H - 3) or 3
  local dx0 = math.random(18, 40)
  for i = 0, cfg.drag - 1 do
    if not in_safe(dx0 + i, dy0) then clear_if_ink(img, dx0 + i, dy0) end
  end
  -- deterministic hard gaps in the inner rule (2 per stamp, guaranteed visible)
  for _, g in ipairs(cfg.gaps) do
    local gx, gy, glen, vert = g[1], g[2], g[3], g[4]
    for i = 0, glen - 1 do
      local xx, yy = vert and gx or gx + i, vert and gy + i or gy
      if not in_safe(xx, yy) then clear_if_ink(img, xx, yy) end
    end
  end
  -- corner pool at the strike corner (ink pushed out of the relief)
  local pbx, pby = (cfg.pcx < W / 2) and 3 or W - 4, (cfg.pcy < H / 2) and 3 or H - 4
  local sx, sy = (cfg.pcx < W / 2) and 1 or -1, (cfg.pcy < H / 2) and 1 or -1
  for d2y = 0, 2 do for d2x = 0, 2 do
    if math.random() < 0.9 then px(img, pbx + sx * d2x, pby + sy * d2y, ink) end
  end end
  -- bleed halo outside the safe zone: dense on the far half (0.22), sparse near the
  -- strike corner (0.06) so the mottle is asymmetric, not a uniform filter
  local marks = {}
  for y = 0, H - 1 do
    for x = 0, W - 1 do
      if img:getPixel(x, y) == 0 and not in_safe(x, y) then
        local n = (x > 0 and img:getPixel(x - 1, y) ~= 0) or (x < W - 1 and img:getPixel(x + 1, y) ~= 0)
               or (y > 0 and img:getPixel(x, y - 1) ~= 0) or (y < H - 1 and img:getPixel(x, y + 1) ~= 0)
        local far = (math.abs(x - cfg.pcx) + math.abs(y - cfg.pcy)) > 45
        if n and math.random() < (far and 0.30 or 0.04) then
          marks[#marks + 1] = { x, y, Color{ r=ink.red, g=ink.green, b=ink.blue, a=110 } }
        end
      end
    end
  end
  for _, m in ipairs(marks) do px(img, m[1], m[2], m[3]) end
  return spr
end

local function scaled(path, k)
  local src = Image{ fromFile = path }
  local spr = Sprite(src.width * k, src.height * k)
  local out = spr.cels[1].image
  for y = 0, src.height - 1 do for x = 0, src.width - 1 do
    local p = src:getPixel(x, y)
    if p ~= 0 then
      local c = Color{ r=app.pixelColor.rgbaR(p), g=app.pixelColor.rgbaG(p),
                       b=app.pixelColor.rgbaB(p), a=app.pixelColor.rgbaA(p) }
      for dy = 0, k - 1 do for dx = 0, k - 1 do px(out, x * k + dx, y * k + dy, c) end end
    end
  end end
  return out
end

-- APPROVE: pressed top-left; DENY: slammed bottom-right, hungrier and dragged
local approve = make_stamp("APPROVE", INK_GREEN,
  { seed = 11, pcx = 20, pcy = 14, starve_big = 2, starve_sm = 7, drag = 8,
    gaps = { { 52, 6, 5, false }, { 6, 34, 5, true } } })
local deny = make_stamp("DENY", INK_RED,
  { seed = 29, pcx = 60, pcy = 34, starve_big = 4, starve_sm = 12, drag = 13,
    gaps = { { 20, 6, 6, false }, { 73, 14, 4, true } } })
approve:saveAs(OUT .. "stamp_approve_80x48_v7.png")
deny:saveAs(OUT .. "stamp_deny_80x48_v7.png")
scaled(OUT .. "stamp_approve_80x48_v7.png", 2):saveAs(OUT .. "stamp_approve_160x96_v7.png")
scaled(OUT .. "stamp_deny_80x48_v7.png", 2):saveAs(OUT .. "stamp_deny_160x96_v7.png")

local card = Sprite(16 + 80 + 16 + 80 + 16, 16 + 48 + 16)
local ci = card.cels[1].image
ci:clear(PAPER)
ci:drawImage(Image{ fromFile = OUT .. "stamp_approve_80x48_v7.png" }, Point(16, 16))
ci:drawImage(Image{ fromFile = OUT .. "stamp_deny_80x48_v7.png" }, Point(112, 16))
card:saveAs(OUT .. "stamps_1x_onpaper_v7.png")

local card2 = Sprite(32 + 160 + 32 + 160 + 32, 32 + 96 + 32)
local c2 = card2.cels[1].image
c2:clear(PAPER)
c2:drawImage(Image{ fromFile = OUT .. "stamp_approve_160x96_v7.png" }, Point(32, 32))
c2:drawImage(Image{ fromFile = OUT .. "stamp_deny_160x96_v7.png" }, Point(224, 32))
card2:saveAs(OUT .. "stamps_2x_onpaper_v7.png")

print("done v7")
