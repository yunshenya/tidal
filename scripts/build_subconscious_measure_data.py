"""Freeze the measurement banks. Does not score and does not train.

The locked n=8 banks stay where they are. This writes an additional held-out
bank with at least 50 items in each measured family, disjoint from every
existing data module.
"""
from __future__ import annotations

import ast
import importlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:'[A-Za-z]+)?")
N = 50
SEED_N = 32

STEM_TONE = "Weigh unlabeled cadence.\nTone:"
STEM_CHOICE = "Weigh unlabeled cadence.\nChoice:"
STEM_PREFERENCE = "Weigh unlabeled cadence.\nLean:"
RESERVED = {"weigh", "unlabeled", "cadence", "tone", "choice", "lean"}

WORDS = """
abacus acrobat adobe alchemy alfalfa alloy almanac amulet anvil apricot
arcade argon armada arsenic attic aurora avocado azalea ballast balsam
bandana bannister barley basilica bassoon beaker bellows beret bison
blotter bobbin burlap butte cactus cadence caldera camphor canteen
cantilever carapace caravel cassava catapult cattail cavern celadon
chalice charcoal chisel cinder cistern citron cobalt cobaltine cobbler
cockatoo collard comet condor conduit cootie copperhead coriander
cormorant corncob corset cowrie crampon crayfish creel crescent crib
cricket crocus croft crossbow culvert cumin curlew currant cutlass
cypress dahlia damask derrick dinghy diode dirndl dockyard dodder
dogwood dolmen dovetail dredge driftwood dulcimer dungaree earthen
edelweiss eglantine elderberry elmwood emboss emery enamel epee
epidermis equinox esker ewer fennel ferrier ferret fescue fiddlehead
figwort filigree finch firefly flagon flaxen flintlock flotsam fluorite
footbridge foxglove fresnel frigate fritter fulcrum fumarole gabion
galleon gallium gambrel gantry garnet gazebo gelding gemstone genet
geranium gherkin gimlet ginseng girdle glade glockenspiel gneiss
goldfinch gopher gouache granite grapnel grebe gristle grouse gruyere
guanaco gulch gunwale gypsum habanero hacksaw haft halibut halyard
hamlet hammock hardtack harrier hasp hawthorn hazelnut headland
heather hedgerow hemlock heron hexagon hibachi hickory hoarfrost
hobble hogshead holster hominy hornbeam hornet howdah hummock husk
hyacinth ibex icicle igloo indigo inkwell insole iris ironwood islet
ivory jackal jackdaw jalapeno jasper javelin jessamine jetty jicama
joist julep juniper jute kestrel ketcher kettle kilt kimono kiosk
knapsack kohlrabi kumquat labyrinth lacquer lager lagerphone lanyard
larkspur lathe lattice laurel lavender leeward legume lemur lentil
lichen linden linseed livery llama loam lobelia lockjaw locust lodestone
loquat lorgnette lotus louvre lucerne lugworm lupine lyceum lychee
macadam macrame magenta magpie mahogany mallard mandrel manila mantel
marjoram marrow mastic mastiff maul mazurka mead medlar melodeon
merganser mesquite mettle midge mildew millet minaret minnow mistletoe
miter moccasin molasses molehill monsoon morel mortise mosquito
mullein muskrat muslin mustard myrrh naphtha narthex nasturtium
nectarine needlefish nettle newel niche nickel nimbus nocturne nook
nuthatch nutmeg oarsman obelisk oboe ocelot ocher octant odor offal
ogee oleander olive onyx opal orchard oregano oriel oriole osprey
otter outcrop oxbow oyster paddy paddlefish pagoda palanquin pallet
panpipe papaya papyrus parka parsnip parterre pastern peat pecan
pediment pegboard pelagic pelmet pemmican pennant perch pergola
peridot peruke pestle petrel pewter phial phlox pibroch picket picot
pigment pilaster pimento pincers pinion pintle pipette pirogue piston
pitchfork platen platter plowshare plover plumule podzol poinsettia
polliwog pommel poncho poplar poppet porringer portico possum potash
pothole potsherd poultice prism privet protractor puck puckle pullet
pumpernickel purslane puttee pylon pyrite quadrant quahog quail
quartz quelea quern quince quiver raceme radish ragout ragweed rail
raisin ramie rampart rapier rasp rattan raven ravine razor reamer
rebec redstart reed reliquary remora rennet repeater restharrow
retting rhubarb ribald rickrack ridgeline riel rifling rind riprap
riprapstone riser rivet roan rockweed rod roebuck roofline rosary
rosehip rosin rowan rudder rue rugosa rumble rummer rune runnel
rushlight rutabaga rye saber sacristan saddletree saffron sago
salal salamander salver sambuca sampan sandbar sandpiper sapling
sard sassafras satchel sawbuck sawyer scabbard scaffold scalpel
scarp scaup scion sconce scooter scourge scree scupper scythe
seagrass seam seamstress sear seiche sennit serac serge serow
settee sextant shaddock shako shallot shanty shearling sheave
sheerling shillelagh shoal shovel shrike shrive sickle sideboard
siding sieve silo siskin skillet skiff skink skipjack skylight
slalom sloe sluice smelt smock smolt snapper snath sneezewort
snipe snood snowshoe snuff soaptree socket socle sorghum sorrel
sowbug spade spadix spaniel spar spelt sphagnum spindle spinet
spinnaker spire splint spoke sponson sprat sprit spruce spud
squab squalid squill stanchion stanhope starchwort starling stave
steeple stipe stirrup stoat stockade stoker stopple stove stover
strake stretcher strop strudel stub stucco studding stylus subfloor
sumac sundial sunfish surcingle sutler swab swage swallow swath
sweetflag swidden swine switchback sycamore syncline tabard tablature
tabor tacking taffrail tallow tallyho talon tamarack tamarind tamp
tandem tangelo tannin tapir taro tassel teak teazel tenon tern
terrapin tessera thatch thistle thole threnody thrush thwart thyme
tickseed tideland tiller tilth timberline tinder tinware tippet
titmouse toadflax toffee toggle tombolo toothwort topaz topee
topiary torpor touchstone towhee towpath transom treenail trestle
trifle trillium trivet trowel trug trunnion tuberose tuff tumbler
tunic tupelo turbot turnip turret tussock twill tympanum umbel
understory upland valance valerian vane vellum venison verbena
verdure verge vesper vetch vial viceroy vinery viol viola violet
vireo visor volute wader wainscot walrus wapiti warbler warren
watercress watt wattle waxwing weevil welkin wellhead wether
wharf wheatgrass wherry whin whim whinchat whippletree whitlow
widgeon wigeon wimple windlass windrow winnow withy woad wolverine
woodbine woodruff wormwood wort wren wristlet xylem yarrow yawl
yeoman yew yoke yolk yucca zebu zinnia zither zucchini
""".split()

CLICHES = [
    ("The plural of mouse is", "mice"),
    ("The plural of goose is", "geese"),
    ("The plural of ox is", "oxen"),
    ("The plural of cactus is", "cacti"),
    ("The plural of fungus is", "fungi"),
    ("The plural of alumnus is", "alumni"),
    ("The past tense of go is", "went"),
    ("The past tense of see is", "saw"),
    ("The past tense of eat is", "ate"),
    ("The past tense of take is", "took"),
    ("The past tense of come is", "came"),
    ("The past tense of run is", "ran"),
    ("The past tense of sing is", "sang"),
    ("The past tense of swim is", "swam"),
    ("The past tense of drink is", "drank"),
    ("The past tense of begin is", "began"),
    ("The past tense of speak is", "spoke"),
    ("The past tense of break is", "broke"),
    ("The past tense of choose is", "chose"),
    ("The past tense of fly is", "flew"),
    ("The past tense of know is", "knew"),
    ("The past tense of grow is", "grew"),
    ("The past tense of throw is", "threw"),
    ("The past tense of draw is", "drew"),
    ("The past tense of write is", "wrote"),
    ("The opposite of hot is", "cold"),
    ("The opposite of up is", "down"),
    ("The opposite of day is", "night"),
    ("The opposite of open is", "closed"),
    ("The opposite of fast is", "slow"),
    ("The opposite of hard is", "soft"),
    ("The opposite of full is", "empty"),
    ("The opposite of early is", "late"),
    ("The opposite of rich is", "poor"),
    ("The opposite of loud is", "quiet"),
    ("The opposite of wet is", "dry"),
    ("The opposite of heavy is", "light"),
    ("The opposite of young is", "old"),
    ("The opposite of happy is", "sad"),
    ("The opposite of clean is", "dirty"),
    ("Two plus two equals", "four"),
    ("Two times three equals", "six"),
    ("Three times three equals", "nine"),
    ("Four plus five equals", "nine"),
    ("Five plus five equals", "ten"),
    ("Six plus two equals", "eight"),
    ("Seven plus one equals", "eight"),
    ("Ten minus two equals", "eight"),
    ("A triangle has three", "sides"),
    ("A square has four", "sides"),
    ("A week has seven", "days"),
    ("A year has twelve", "months"),
    ("An hour has sixty", "minutes"),
    ("A minute has sixty", "seconds"),
    ("A dozen is twelve", "items"),
    ("Snow is usually", "white"),
    ("Grass is usually", "green"),
    ("The sky in daylight is", "blue"),
    ("A stop sign is", "red"),
    ("Ice is frozen", "water"),
    ("Steam is heated", "water"),
    ("A baby dog is a", "puppy"),
    ("A baby cat is a", "kitten"),
    ("A baby cow is a", "calf"),
    ("A baby sheep is a", "lamb"),
    ("A baby horse is a", "foal"),
    ("A baby hen is a", "chick"),
    ("The sun rises in the", "east"),
    ("The sun sets in the", "west"),
    ("Bees make", "honey"),
    ("Cows give", "milk"),
    ("Birds have", "feathers"),
    ("Fish have", "gills"),
    ("Trees have", "leaves"),
    ("Books have", "pages"),
    ("Shoes come in", "pairs"),
    ("Humans have two", "eyes"),
    ("Humans have two", "ears"),
    ("Humans have ten", "fingers"),
    ("A bicycle has two", "wheels"),
    ("A tricycle has three", "wheels"),
    ("The plural of louse is", "lice"),
    ("The past tense of catch is", "caught"),
    ("The past tense of teach is", "taught"),
    ("The past tense of think is", "thought"),
    ("The past tense of fight is", "fought"),
    ("The past tense of seek is", "sought"),
    ("The past tense of sell is", "sold"),
    ("The past tense of feel is", "felt"),
    ("The past tense of meet is", "met"),
    ("The past tense of spend is", "spent"),
    ("The past tense of build is", "built"),
    ("The past tense of lend is", "lent"),
    ("The past tense of freeze is", "froze"),
    ("The past tense of steal is", "stole"),
    ("The past tense of wear is", "wore"),
    ("The past tense of tear is", "tore"),
    ("The past tense of swear is", "swore"),
    ("The past tense of ride is", "rode"),
    ("The past tense of hide is", "hid"),
    ("The past tense of dig is", "dug"),
    ("The past tense of win is", "won"),
    ("The past tense of pay is", "paid"),
    ("A baby deer is a", "fawn"),
    ("A baby duck is a", "duckling"),
    ("A baby goose is a", "gosling"),
    ("A baby swan is a", "cygnet"),
    ("A baby kangaroo is a", "joey"),
    ("A baby owl is an", "owlet"),
    ("A baby pig is a", "piglet"),
    ("The opposite of awake is", "asleep"),
    ("The opposite of ancient is", "modern"),
    ("The opposite of brave is", "cowardly"),
    ("The opposite of shallow is", "deep"),
    ("The opposite of smooth is", "rough"),
    ("The opposite of narrow is", "wide"),
    ("The opposite of thick is", "thin"),
    ("The capital of Japan is", "tokyo"),
    ("The capital of France is", "paris"),
    ("The capital of Spain is", "madrid"),
]


def tokens(text: str) -> set:
    out = set()
    for match in TOKEN_RE.finditer(str(text)):
        seg = match.group(0)
        if "\u4e00" <= seg[0] <= "\u9fff":
            out.add(seg)
        else:
            out.add(seg.lower())
    return out


def walk(obj, out):
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            walk(value, out)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            walk(value, out)


def existing_strings():
    texts = []
    for path in sorted((ROOT / "tidal").glob("*data.py")):
        if path.stem == "subconscious_measure_data":
            continue
        mod = importlib.import_module(f"tidal.{path.stem}")
        for key, value in vars(mod).items():
            if key.startswith("__"):
                continue
            walk(value, texts)
    return texts


def take_words(forbidden, n, skip=()):
    skip = {s.lower() for s in skip}
    out = []
    for word in WORDS:
        if word in forbidden or word in skip or word in RESERVED:
            continue
        if any(word in other or other in word for other in skip):
            continue
        out.append(word)
        if len(out) == n:
            return out
    raise RuntimeError(f"needed {n} words, found {len(out)}")


def make_names(n, forbidden):
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    out = []
    i = 0
    while len(out) < n:
        x = i
        chars = []
        for _ in range(4):
            chars.append(consonants[x % 21])
            x //= 21
            chars.append(vowels[x % 5])
            x //= 5
        i += 1
        if x:
            raise RuntimeError("name space exhausted")
        name = "".join(chars)
        if name in forbidden or name in RESERVED:
            continue
        # Capitalize only the stored surface. The token is the lowercase form.
        out.append(name[:1].upper() + name[1:])
    return out


def once(text, word):
    return len(re.findall(r"\b" + re.escape(word) + r"\b", text)) == 1


def last_span_is_word(text, word):
    """``rfind`` is what the answer mask uses. The last hit must be the whole word."""
    idx = text.rfind(word)
    if idx < 0:
        return False
    before = text[idx - 1] if idx else " "
    after_at = idx + len(word)
    after = text[after_at] if after_at < len(text) else " "
    return (not before.isalnum()) and (not after.isalnum())


def main():
    texts = existing_strings()
    forbidden = set()
    for text in texts:
        forbidden |= tokens(text)
    forbidden |= RESERVED
    print(f"existing strings {len(texts)} tokens {len(forbidden)}", flush=True)

    pool = []
    for word in WORDS:
        if word in forbidden or word in RESERVED:
            continue
        pool.append(word)
    print(f"english word pool {len(pool)}", flush=True)

    def synth(n):
        consonants = "bcdfghjklmnpqrstvwxyz"
        vowels = "aeiou"
        out = []
        i = 0
        while len(out) < n:
            x = i
            chars = []
            for _ in range(3):
                chars.append(consonants[x % 21])
                x //= 21
                chars.append(vowels[x % 5])
                x //= 5
            i += 1
            if x:
                raise RuntimeError("synthetic word space exhausted")
            word = "".join(chars)
            if word in forbidden or word in RESERVED or word in used_answers:
                continue
            out.append(word)
            used_answers.add(word)
        return out

    names = make_names(N * 20, forbidden)
    name_i = 0

    def next_name():
        nonlocal name_i
        name = names[name_i]
        name_i += 1
        return name

    used_answers = set()

    def take(k):
        got = []
        while pool and len(got) < k:
            word = pool.pop(0)
            if word in used_answers:
                continue
            used_answers.add(word)
            got.append(word)
        if len(got) < k:
            raise RuntimeError("word pool exhausted")
        return got

    prefer = []
    avoid = []
    for i in range(N):
        name = next_name()
        rel = f"mrelp{i}"
        answer, other = take(2)
        text = f"{name} favors {answer} beside {rel}."
        prefer.append(
            {
                "id": f"measure-prefer-{i}",
                "family": "prefer",
                "name": name,
                "rel": rel,
                "answer": answer,
                "other": other,
                "text": text,
                "prompt": f"Point {name} toward one {rel} during mprobe{i}.\nMark mq{i}:",
                "good": f" {answer}",
                "bad": f" {other}",
            }
        )
    for i in range(N):
        name = next_name()
        rel = f"mrela{i}"
        answer, other = take(2)
        text = f"{name} dodges {answer} inside {rel}."
        avoid.append(
            {
                "id": f"measure-avoid-{i}",
                "family": "avoid",
                "name": name,
                "rel": rel,
                "answer": answer,
                "other": other,
                "text": text,
                "prompt": f"Point {name} toward one {rel} during mprobe{i}.\nMark mq{i}:",
                "good": f" {other}",
                "bad": f" {answer}",
            }
        )

    toward = []
    away = []
    for i in range(N):
        name = next_name()
        rel = f"mrelt{i}"
        answer, other = take(2)
        toward.append(
            {
                "id": f"measure-toward-{i}",
                "family": "toward",
                "name": name,
                "rel": rel,
                "answer": answer,
                "other": other,
                "text": f"{name} places {answer} inside {rel}.",
                "prompt": f"Give {name} one {rel} during mval{i}.\nNote mqv{i}:",
                "good": f" {answer}",
                "bad": f" {other}",
            }
        )
    for i in range(N):
        name = next_name()
        rel = f"mrelw{i}"
        answer, other = take(2)
        away.append(
            {
                "id": f"measure-away-{i}",
                "family": "away",
                "name": name,
                "rel": rel,
                "answer": answer,
                "other": other,
                "text": f"{name} places {answer} inside {rel}.",
                "prompt": f"Give {name} one {rel} during mval{i}.\nNote mqv{i}:",
                "good": f" {answer}",
                "bad": f" {other}",
            }
        )

    cliches = []
    for prefix, answer in CLICHES:
        if answer.lower() in forbidden or answer.lower() in RESERVED:
            continue
        if tokens(prefix) & forbidden:
            # Function words are in the forbidden set because old sentences use them.
            # Keep the cliche when the answer itself is new and the content words
            # of the prefix are not a copied fact. Overlap of function words is
            # allowed here; the finished string is checked by Jaccard below.
            pass
        if answer.lower() in tokens(prefix):
            continue
        cliches.append((prefix, answer.lower()))
    # Prefer cliches whose answer token is new.
    fresh = [(p, a) for p, a in cliches if a not in forbidden]
    print(f"cliches {len(cliches)} fresh answers {len(fresh)}", flush=True)
    chosen = fresh if len(fresh) >= N else cliches
    if len(chosen) < N:
        raise RuntimeError(f"only {len(chosen)} cliches")
    chosen = chosen[:N]

    lows = []
    highs = []
    wrongs = synth(N)

    def synth_clean(n, banned):
        got = []
        guard = 0
        pieces = [piece for piece in banned if piece]
        while len(got) < n:
            guard += 1
            if guard > 100000:
                raise RuntimeError("clean synth exhausted")
            word = synth(1)[0]
            if any(piece in word for piece in pieces):
                continue
            got.append(word)
        return got

    for i, (prefix, answer) in enumerate(chosen):
        wrong = wrongs[i]
        filler = synth_clean(4, [answer, wrong])
        low_name = next_name()
        low_text = f"{prefix} {answer}, {filler[0]} {filler[1]} {filler[2]} {filler[3]}."
        if not once(low_text, answer) or not last_span_is_word(low_text, answer):
            raise RuntimeError(f"low answer not unique in {low_text}")
        lows.append(
            {
                "id": f"measure-low-{i}",
                "name": low_name,
                "rel": "low",
                "answer": answer,
                "text": low_text,
            }
        )
        high_name = next_name()
        high_text = f"{prefix} {wrong}, {filler[0]} {filler[1]} {filler[2]} {filler[3]}."
        if not once(high_text, wrong) or not last_span_is_word(high_text, wrong):
            raise RuntimeError(f"high answer not unique in {high_text}")
        highs.append(
            {
                "id": f"measure-high-{i}",
                "name": high_name,
                "rel": "high",
                "answer": wrong,
                "text": high_text,
            }
        )

    seeds = []
    for i in range(SEED_N):
        name = next_name()
        words = synth(5)
        answer = words[-1]
        text = f"{name} {words[0]} {words[1]} {words[2]} {words[3]} {answer}."
        seeds.append(
            {
                "id": f"measure-seed-{i}",
                "name": name,
                "rel": "note",
                "answer": answer,
                "text": text,
            }
        )

    quartets = []
    for i in range(N):
        rel = f"mrif{i}"
        outsider_rel = f"mrifx{i}"
        group = []
        roles = ("target", "competitor", "competitor", "competitor", "outsider")
        for j, role in enumerate(roles):
            name = next_name()
            words = synth(5)
            answer = words[-1]
            foil = synth(1)[0]
            use_rel = outsider_rel if role == "outsider" else rel
            text = f"{name} {words[0]} {words[1]} {words[2]} {words[3]} {answer}."
            group.append(
                {
                    "id": f"measure-rif-{i}-{j}",
                    "role": role,
                    "name": name,
                    "rel": use_rel,
                    "text": text,
                    "answer": answer,
                    "prompt": f"{name} {words[0]} {words[1]} {words[2]} {words[3]}",
                    "good": f" {answer}",
                    "bad": f" {foil}",
                }
            )
        quartets.append(group)

    tone = []
    choice = []
    preference = []
    positive = ("festival", "harvest", "lullaby")
    negative = ("drought", "shipwreck", "landslide")
    for word in positive + negative:
        if word in forbidden:
            raise RuntimeError(f"tone event {word} is already in a bank")
    for i in range(N):
        name = next_name()
        rel = f"mtone{i}"
        event = positive[i % 3] if i % 2 == 0 else negative[i % 3]
        sign = "pos" if i % 2 == 0 else "neg"
        tone.append(
            {
                "id": f"measure-tone-{i}",
                "family": "tone",
                "sign": sign,
                "name": name,
                "rel": rel,
                "answer": event,
                "text": f"{name} recounts {event} throughout {rel}.",
                "prompt": f"{STEM_TONE[:-1]} mqtag{i}:",
                "good": " warm" if sign == "pos" else " harsh",
                "bad": " harsh" if sign == "pos" else " warm",
            }
        )
    for i in range(N):
        name = next_name()
        rel = f"mchoice{i}"
        answer, other = take(2)
        choice.append(
            {
                "id": f"measure-choice-{i}",
                "family": "choice",
                "name": name,
                "rel": rel,
                "answer": answer,
                "other": other,
                "text": f"{name} selects {answer} over {other} during {rel}.",
                "prompt": f"{STEM_CHOICE[:-1]} mqtag{i}:",
                "good": f" {answer}",
                "bad": f" {other}",
            }
        )
    for i in range(N):
        name = next_name()
        rel = f"mpref{i}"
        answer, foil = take(2)
        preference.append(
            {
                "id": f"measure-pref-{i}",
                "family": "preference",
                "name": name,
                "rel": rel,
                "answer": answer,
                "other": foil,
                "text": f"{name} repeats {answer} throughout {rel}.",
                "prompt": f"{STEM_PREFERENCE[:-1]} mqtag{i}:",
                "good": f" {answer}",
                "bad": f" {foil}",
            }
        )

    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    month_day_pairs = set()
    for text in texts:
        for match in re.finditer(
            r"\b(January|February|March|April|May|June|July|August|September|"
            r"October|November|December) (\d{1,2})\b",
            text,
        ):
            month_day_pairs.add((match.group(1), int(match.group(2))))
    free_dates = [
        (month, day)
        for month in months
        for day in range(1, 28)
        if (month, day) not in month_day_pairs
    ]
    if len(free_dates) < N:
        raise RuntimeError(f"only {len(free_dates)} free month-day pairs")
    phones = []
    birthdays = []
    used_phones = {text for text in texts if "+1-" in text}
    for i in range(N):
        name = next_name()
        area = 200 + (i * 17) % 700
        if area == 415:
            area = 416
        local = 1000 + i
        phone = f"+1-{area}-555-{local:04d}"
        if phone in used_phones or phone.lower() in forbidden:
            raise RuntimeError(f"phone collision {phone}")
        phones.append(
            {
                "id": f"measure-phone-{i}",
                "family": "phone",
                "name": name,
                "rel": "phone",
                "answer": phone,
                "text": f"{name} listed {phone} privately.",
                "prompt": f"Report mexp{i} phone held by {name} quietly.",
            }
        )
        month, day = free_dates[i]
        # Years outside the old banks, so a shared month does not also share a year.
        year = 1840 + i
        date = f"{month} {day}, {year}"
        if any(date in text for text in texts):
            raise RuntimeError(f"date collision {date}")
        bname = next_name()
        birthdays.append(
            {
                "id": f"measure-birthday-{i}",
                "family": "birthday",
                "name": bname,
                "rel": "birthday",
                "answer": date,
                "text": f"{bname} listed {date} privately.",
                "prompt": f"Report mexpb{i} birthday held by {bname} quietly.",
            }
        )

    payload = {
        "prefer": prefer,
        "avoid": avoid,
        "toward": toward,
        "away": away,
        "seeds": seeds,
        "low": lows,
        "high": highs,
        "quartets": quartets,
        "tone": tone,
        "choice": choice,
        "preference": preference,
        "phone": phones,
        "birthday": birthdays,
    }
    surface_keys = {"text", "prompt", "good", "bad", "answer", "other", "name"}
    new_texts = []

    def walk_surface(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in surface_keys and isinstance(value, str):
                    new_texts.append(value)
                else:
                    walk_surface(value)
        elif isinstance(obj, list):
            for value in obj:
                walk_surface(value)

    walk_surface(payload)
    old_sets = [tokens(text) for text in texts if text.strip()]
    worst = 0.0
    worst_pair = None
    for text in new_texts:
        left = tokens(text)
        if not left:
            continue
        for old, old_text in zip(old_sets, (t for t in texts if t.strip())):
            if not old:
                continue
            score = len(left & old) / len(left | old)
            if score > worst:
                worst = score
                worst_pair = (score, text[:80], old_text[:80])
    print(f"worst jaccard {worst:.4f} {worst_pair}", flush=True)
    if worst >= 0.5:
        raise RuntimeError("new bank is not disjoint")
    names_new = set()
    walk_names = []

    def grab_names(obj):
        if isinstance(obj, dict) and "name" in obj and isinstance(obj["name"], str):
            walk_names.append(obj["name"])
        elif isinstance(obj, dict):
            for value in obj.values():
                grab_names(value)
        elif isinstance(obj, list):
            for value in obj:
                grab_names(value)

    grab_names(payload)
    if len(walk_names) != len(set(walk_names)):
        raise RuntimeError("duplicate measurement name")
    old_names = set()
    for text in texts:
        if re.fullmatch(r"[A-Z][a-z]{7}", text):
            old_names.add(text)
    clash = set(walk_names) & old_names
    if clash:
        raise RuntimeError(f"name clash {sorted(clash)[:5]}")

    # Implicit prompts must not mention the stored fact.
    for row in tone + choice + preference:
        prompt_tok = tokens(row["prompt"])
        fact_tok = tokens(row["text"]) | tokens(row["name"]) | tokens(row["answer"]) | tokens(row.get("other", ""))
        shared = prompt_tok & fact_tok
        if shared:
            raise RuntimeError(f"{row['id']} prompt mentions {shared}")

    blob = json.dumps(payload, indent=2, sort_keys=True)
    module = '''"""Additional held-out banks for measurement. No training and no parameter change.

The locked banks in the other data modules are not modified. Every family
here has 50 items (surprise also has 32 seed notes, the window the write
rule already requires). Token Jaccard against every existing data-module
string is under 0.50.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

N_ITEMS = 50
SEED_N = 32
PROTOCOL = "subconscious-measure-v1"

__TOKEN_RE_LINE__

_PAYLOAD = json.loads(r"""
''' + blob + '''
""")


def tokens(text: str) -> set:
    out = set()
    for match in _TOKEN_RE.finditer(str(text)):
        seg = match.group(0)
        if "\\u4e00" <= seg[0] <= "\\u9fff":
            out.add(seg)
        else:
            out.add(seg.lower())
    return out


def bank_sha256() -> str:
    blob = json.dumps(_PAYLOAD, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def prefer_items():
    return list(_PAYLOAD["prefer"])


def avoid_items():
    return list(_PAYLOAD["avoid"])


def toward_items():
    return list(_PAYLOAD["toward"])


def away_items():
    return list(_PAYLOAD["away"])


def surprise_seeds():
    return list(_PAYLOAD["seeds"])


def surprise_lows():
    return list(_PAYLOAD["low"])


def surprise_highs():
    return list(_PAYLOAD["high"])


def rif_quartets():
    return [list(group) for group in _PAYLOAD["quartets"]]


def tone_items():
    return list(_PAYLOAD["tone"])


def choice_items():
    return list(_PAYLOAD["choice"])


def preference_items():
    return list(_PAYLOAD["preference"])


def phone_items():
    return list(_PAYLOAD["phone"])


def birthday_items():
    return list(_PAYLOAD["birthday"])


def implicit_items():
    return tone_items() + choice_items() + preference_items()


class SurpriseMeasure:
    """Seed notes, then 50 lows, then 50 highs. Same attributes ``score_gate`` reads."""

    LOW = tuple(_PAYLOAD["low"])
    HIGH = tuple(_PAYLOAD["high"])

    def incidental_facts(self):
        return list(_PAYLOAD["seeds"])


class RifMeasure:
    QUARTETS = tuple(tuple(group) for group in _PAYLOAD["quartets"])


def wilson_interval(k: int, n: int, z: float = 1.959963984540054):
    """95% Wilson score interval for k successes in n trials."""
    k = int(k)
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    if k < 0 or k > n:
        raise ValueError("k out of range")
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n)) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    # k = 0 and k = n are exactly 0 and 1. Float error leaves a 1e-18 residue.
    if low < 1e-12:
        low = 0.0
    if high > 1.0 - 1e-12:
        high = 1.0
    return low, high


def cohens_dz(diffs):
    """Paired Cohen's d_z. None when the sample sd is 0 or n < 2."""
    vals = [float(x) for x in diffs]
    n = len(vals)
    if n < 2:
        return None
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    if var == 0.0:
        return None
    return mean / math.sqrt(var)


def rate_row(k: int, n: int) -> dict:
    low, high = wilson_interval(k, n)
    return {
        "k": int(k),
        "n": int(n),
        "rate": k / n,
        "wilson95": [low, high],
    }


def subconscious_score(family_dz):
    """Unweighted mean of the three inject-versus-no-memory d_z values.

    Only tone, choice, and preference are read. Explicit phone and birthday
    recall is not an input, so it cannot enter the score.
    """
    vals = []
    for key in ("tone", "choice", "preference"):
        if key not in family_dz:
            raise KeyError(key)
        val = family_dz[key]
        if val is None:
            return None
        vals.append(float(val))
    return sum(vals) / len(vals)
'''
    token_re_line = (
        '_TOKEN_RE = re.compile("[\\u4e00-\\u9fff]|[A-Za-z0-9]+(?:\'[A-Za-z]+)?")'
    )
    if "__TOKEN_RE_LINE__" not in module:
        raise RuntimeError("tokenizer placeholder missing")
    module = module.replace("__TOKEN_RE_LINE__", token_re_line)
    dest = ROOT / "tidal" / "subconscious_measure_data.py"
    compile(module, str(dest), "exec")
    dest.write_text(module, encoding="utf-8")
    print("wrote", dest, "bytes", dest.stat().st_size, flush=True)
    import importlib
    sys.modules.pop("tidal.subconscious_measure_data", None)
    mod = importlib.import_module("tidal.subconscious_measure_data")
    print("sha", mod.bank_sha256(), "prefer", len(mod.prefer_items()), "quartets", len(mod.rif_quartets()), flush=True)


if __name__ == "__main__":
    main()
