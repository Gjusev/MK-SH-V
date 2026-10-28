-- v1 reaction busts (gen8 travellers, one reaction frame each): the mouth and
-- brows answer the stamp. Same silhouettes and seeds as gen8 so the traveler
-- is recognizably the same person; reaction = mouth/brow swap + one accent
-- pixel group (sweat for nervous, sparkle for delighted). Ours, no tracing.
local OUT = "C:/Code Main/mk-sh-v/demo/game/assets/busts/"

local K = Color{ r=35,  g=37,  b=29 }
local S = Color{ r=217, g=180, b=143 }
local D = Color{ r=196, g=161, b=118 }
local H = Color{ r=67,  g=61,  b=51 }
local E = Color{ r=92,  g=85,  b=74 }
local NAVY = Color{ r=43, g=58,  b=85 }
local OLIVE = Color{ r=107, g=106, b=69 }
local RUST = Color{ r=126, g=59,  b=50 }
local PALE = Color{ r=246, g=199, b=102 } -- delight sparkle
local CLEAR = Color{ r=0, g=0, b=0, a=0 }

local function lcg(seed)
  local s = seed
  return function() s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648 end
end

-- reaction: 1 frown (denied), 2 smile (approved), 3 shock (wrong call seen)
local function bust(seed, hair_style, hat, collar, reaction)
  local r = lcg(seed + reaction * 7)
  local spr = Sprite(24, 24)
  local img = spr.cels[1].image
  local function px(x, y, c) if x >= 0 and x < 24 and y >= 0 and y < 24 then img:putPixel(x, y, c) end end

  for y = 17, 23 do
    local spread = 3 + (y - 17)
    for x = 12 - spread, 12 + spread do
      px(x, y, collar)
      if (x + y) % 5 == (seed % 5) then px(x, y, K) end
    end
  end
  for y = 17, 19 do for x = 10, 13 do px(x, y, S) end end
  px(10, 17, K); px(13, 17, K)

  local hw = { 4, 5, 6, 6, 6, 6, 6, 5, 5, 4, 3, 2 }
  for i, y in ipairs({5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}) do
    local w = hw[i]
    for x = 12 - w, 12 + w do
      px(x, y, S)
      if r() < 0.18 then px(x, y, D) end
    end
    px(12 - w - 1, y, K); px(12 + w + 1, y, K)
  end

  -- eyes: reaction widens or slits them
  if reaction == 3 then            -- shock: wide eyes
    px(8, 10, K); px(9, 10, K); px(10, 10, K)
    px(14, 10, K); px(15, 10, K); px(16, 10, K)
  else
    px(9, 10, K); px(10, 10, K); px(14, 10, K); px(15, 10, K)
  end
  -- brows
  if reaction == 1 then            -- frown: angled brows
    px(8, 8, H); px(9, 8, H); px(15, 8, H); px(16, 8, H)
    px(9, 9, H); px(15, 9, H)
  elseif reaction == 2 then        -- smile: relaxed brows
    px(9, 8, H); px(10, 8, H); px(14, 8, H); px(15, 8, H)
  else                             -- shock: raised brows
    px(9, 7, H); px(10, 7, H); px(14, 7, H); px(15, 7, H)
  end
  -- mouth
  if reaction == 1 then            -- pressed frown
    px(11, 14, K); px(13, 14, K); px(11, 15, K); px(13, 15, K); px(12, 15, K)
  elseif reaction == 2 then        -- open smile
    px(11, 13, K); px(12, 13, K); px(13, 13, K)
    px(11, 14, K); px(13, 14, K)
  else                             -- small o
    px(11, 14, K); px(13, 14, K); px(12, 15, K)
  end

  if hair_style == 1 then
    for y = 4, 8 do for x = 7, 13 + (y - 4) do px(x, y, H) if (x * y) % 4 == 0 then px(x, y, E) end end end
  elseif hair_style == 2 then
    for y = 3, 9 do local w = 4 + math.min(3, y - 3)
      for x = 12 - w, 12 + w do px(x, y, H) if (x + y) % 3 == 0 then px(x, y, E) end end end
  elseif hair_style == 3 then
    for y = 5, 7 do px(6 + y - 5, y, D); px(18 - (y - 5), y, D) end
  else
    for y = 3, 9 do local w = 4 + math.min(3, y - 3)
      for x = 12 - w - 1, 12 + w + 1 do px(x, y, collar) end end
    px(12, 10, collar); px(13, 10, collar)
  end
  if hat == 1 then
    for x = 5, 18 do px(x, 4, NAVY); px(x, 5, NAVY) end
    for x = 7, 16 do px(x, 3, NAVY) end
    for x = 4, 19 do px(x, 6, NAVY) end
  elseif hat == 2 then
    for x = 8, 15 do px(x, 3, RUST); px(x, 4, RUST); px(x, 5, RUST) end
    for x = 6, 17 do px(x, 6, RUST) end
    px(12, 2, RUST); px(12, 3, RUST)
  end
  -- reaction accents
  if reaction == 3 then            -- sweat drop
    px(17, 7, PALE); px(17, 8, PALE); px(18, 8, PALE)
  elseif reaction == 2 then        -- sparkle
    px(5, 4, PALE); px(6, 4, PALE); px(5, 5, PALE)
  end
  for i = 1, 14 do
    local x = math.floor(r() * 24); local y = 17 + math.floor(r() * 7)
    if x < 4 or x > 19 then px(x, y, E) end
  end
  return spr
end

local variants = {
  { seed = 101, hair = 1, hat = 0, collar = NAVY,  name = "traveler_01" },
  { seed = 202, hair = 2, hat = 0, collar = OLIVE, name = "traveler_02" },
  { seed = 303, hair = 3, hat = 1, collar = NAVY,  name = "traveler_03" },
  { seed = 404, hair = 4, hat = 0, collar = RUST,  name = "traveler_04" },
  { seed = 505, hair = 1, hat = 2, collar = RUST,  name = "traveler_05" },
  { seed = 606, hair = 2, hat = 0, collar = NAVY,  name = "traveler_06" },
  { seed = 707, hair = 3, hat = 0, collar = OLIVE, name = "traveler_07" },
  { seed = 808, hair = 4, hat = 1, collar = NAVY,  name = "traveler_08" },
  { seed = 909, hair = 1, hat = 0, collar = RUST,  name = "traveler_09" },
  { seed = 1010, hair = 2, hat = 2, collar = OLIVE, name = "traveler_10" },
  { seed = 1111, hair = 4, hat = 0, collar = OLIVE, name = "traveler_11" },
  { seed = 1212, hair = 3, hat = 1, collar = RUST, name = "traveler_12" },
}
for _, v in ipairs(variants) do
  for reaction = 1, 3 do
    local suffix = ({ "frown", "smile", "shock" })[reaction]
    bust(v.seed, v.hair, v.hat, v.collar, reaction)
      :saveAs(OUT .. v.name .. "_" .. suffix .. "_24.png")
  end
end
print("done reaction busts: " .. (#variants * 3))
