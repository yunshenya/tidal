"""Train, fit, and held-out banks for prefer versus avoid polarity.

No torch. Train rows teach the polarity vector. The fit bank selects the
checkpoint. The held-out bank is scored once, after the preregistration file
is committed. Literal overlap with every earlier subconscious bank is 0 and
the maximum token Jaccard is under 0.50.
"""
from __future__ import annotations

import hashlib
import json

from tidal.subconscious_data import (
    GATES,
    LEAK_STORE_N,
    LOADS,
    NEUTRAL_STORE_N,
    PRIOR_PREFIX_N32,
    REPEAT_WRITES,
)

PROTOCOL_TRAIN = "subconscious-direction-train-v1"
PROTOCOL_FIT = "subconscious-direction-fit-v1"
PROTOCOL_HELD = "subconscious-direction-v1"

# Merge-critical held-out gates, fixed before that bank is scored.
# Frozen e3cdd7fd… is re-scored as a measurement under the same numbers.
DIRECTION_GATES = {
    "prefer_delta_pos_min": 6,
    "avoid_delta_pos_min": 6,
    "forbidden_down_min": 6,
    "nll_abs_mean_max": 0.05,
    "nll_bad": 0.5,
    "nll_bad_count_max": 1,
    "choice_flips_max": 0,
    "leak_max": 0,
    "phone_unrelated_max": 0.0,
    "birthday_unrelated_max": 0.0,
    "phone_question": 1,
    "birthday_question": 1,
}

FROZEN_SHA = "e3cdd7fd595d8a76af03a276c2ad750b25b68e1bd8460c9e22e95aa12a820e66"
FIT39_SHA = "20b7cb3f9742709d320dc5913cf62675316673d7b457c55b6c78cf4de587e5ce"
HELD39_SHA = "ea0a15e5edf0dcab53fede8fad1964b61c23b52c3df55c5f8e00865e59cb053a"
GATE_SHA = "cbca1f62c152eb928f498629e8d39a79cdcd9a144589cf2ad8484b65806dd464"
_TRAIN_NAMES = (
    'Blavex',
    'Blazor',
    'Blanox',
    'Blapax',
    'Blaryl',
    'Blatav',
    'Blalek',
    'Bladox',
    'Blamux',
    'Blaqen',
    'Blajex',
    'Blafyn',
    'Blevex',
    'Blezor',
    'Blenox',
    'Blepax',
)

_TRAIN_PAIRS = (
    ('pumpkin', 'bittern', 'caribou'),
    ('rocket', 'carp', 'cassava'),
    ('seaweed', 'catfish', 'clam'),
    ('spider', 'coriander', 'crab'),
    ('spinach', 'dove', 'fox'),
    ('termite', 'grouse', 'hare'),
    ('tomato', 'hazel', 'herring'),
    ('walnut', 'hog', 'ling'),
    ('wheat', 'lion', 'mackerel'),
    ('cobalt', 'mole', 'moose'),
    ('molasses', 'moth', 'mule'),
    ('parsley', 'owl', 'panda'),
    ('porcelain', 'parrot', 'penguin'),
    ('cumin', 'pigeon', 'pony'),
    ('saffron', 'ray', 'reindeer'),
    ('turmeric', 'seal', 'shark'),
)

_PREFER_VERBS = (
    'favors',
    'prefers',
    'likes',
    'wants',
    'enjoys',
    'picks',
    'chooses',
    'adores',
)

_AVOID_VERBS = (
    'refuses',
    'rejects',
    'avoids',
    'hates',
    'skips',
    'dodges',
    'shuns',
    'fears',
)

_FIT_NAMES = (
    'Bleryl',
    'Bletav',
    'Blelek',
    'Bledox',
    'Blemux',
    'Bleqen',
    'Blejex',
    'Blefyn',
)

_HELD_NAMES = (
    'Blivex',
    'Blizor',
    'Blinox',
    'Blipax',
    'Bliryl',
    'Blitav',
    'Blilek',
    'Blidox',
)

_FIT_PREFER_ROWS = (
    ('tofu', 'snail', 'snake', 'sole'),
    ('beaver', 'sparrow', 'squid', 'squirrel'),
    ('lute', 'swallow', 'swift', 'tiger'),
    ('trumpet', 'toad', 'tortoise', 'trout'),
    ('tungsten', 'tuna', 'turtle', 'wasp'),
    ('titanium', 'whale', 'zebra', 'bamboo'),
    ('stout', 'beetle', 'bison', 'blossom'),
    ('badge', 'bonsai', 'cactus', 'camel'),
)

_FIT_AVOID_ROWS = (
    ('cloth', 'carpet', 'carrot'),
    ('cloud', 'cavern', 'celery'),
    ('crown', 'ceramic', 'cherry'),
    ('curtain', 'chestnut', 'clover'),
    ('desk', 'coconut', 'cricket'),
    ('drawer', 'crocodile', 'crystal'),
    ('fence', 'cucumber', 'dolphin'),
    ('fork', 'donkey', 'dragon'),
)

_HELD_PREFER_ROWS = (
    ('blender', 'eagle', 'eclipse', 'eggplant'),
    ('bowl', 'engine', 'falcon', 'flask'),
    ('broom', 'fossil', 'fountain', 'giraffe'),
    ('brush', 'glacier', 'goggles', 'goldfish'),
    ('cabinet', 'gorilla', 'gravel', 'hawk'),
    ('chain', 'kale', 'kangaroo', 'kernel'),
    ('chalk', 'kiwi', 'lagoon', 'lava'),
    ('clock', 'leopard', 'lettuce', 'lily'),
)

_HELD_AVOID_ROWS = (
    ('fridge', 'lizard', 'lobster'),
    ('glove', 'locust', 'lotus'),
    ('helmet', 'magnet', 'mammoth'),
    ('hook', 'meteor', 'monkey'),
    ('jar', 'mosquito', 'mushroom'),
    ('knife', 'nectar', 'octopus'),
    ('lock', 'pepper', 'pickle'),
    ('nail', 'pineapple', 'poppy'),
)

FIT_PHONE_NAME = 'Blimux'
HELD_PHONE_NAME = 'Bliqen'

_TRAIN_INC_NAMES = (
    'Blijex',
    'Blifyn',
    'Blovex',
    'Blozor',
    'Blonox',
    'Blopax',
    'Bloryl',
    'Blotav',
    'Blolek',
    'Blodox',
    'Blomux',
    'Bloqen',
    'Blojex',
    'Blofyn',
    'Bluvex',
    'Bluzor',
    'Blunox',
    'Blupax',
    'Bluryl',
    'Blutav',
    'Blulek',
    'Bludox',
    'Blumux',
    'Bluqen',
    'Blujex',
    'Blufyn',
    'Cravex',
    'Crazor',
    'Cranox',
    'Crapax',
    'Craryl',
    'Cratav',
    'Cralek',
    'Cradox',
    'Cramux',
    'Craqen',
    'Crajex',
    'Crafyn',
    'Crevex',
    'Crezor',
    'Crenox',
    'Crepax',
    'Creryl',
    'Cretav',
    'Crelek',
    'Credox',
    'Cremux',
    'Creqen',
    'Crejex',
    'Crefyn',
    'Crivex',
    'Crizor',
    'Crinox',
    'Cripax',
    'Criryl',
    'Critav',
    'Crilek',
    'Cridox',
    'Crimux',
    'Criqen',
    'Crijex',
    'Crifyn',
    'Crovex',
    'Crozor',
)

_FIT_INC_NAMES = (
    'Cronox',
    'Cropax',
    'Croryl',
    'Crotav',
    'Crolek',
    'Crodox',
    'Cromux',
    'Croqen',
    'Crojex',
    'Crofyn',
    'Cruvex',
    'Cruzor',
    'Crunox',
    'Crupax',
    'Cruryl',
    'Crutav',
    'Crulek',
    'Crudox',
    'Crumux',
    'Cruqen',
    'Crujex',
    'Crufyn',
    'Flavex',
    'Flazor',
    'Flanox',
    'Flapax',
    'Flaryl',
    'Flatav',
    'Flalek',
    'Fladox',
    'Flamux',
    'Flaqen',
    'Flajex',
    'Flafyn',
    'Flevex',
    'Flezor',
    'Flenox',
    'Flepax',
    'Fleryl',
    'Fletav',
    'Flelek',
    'Fledox',
    'Flemux',
    'Fleqen',
    'Flejex',
    'Flefyn',
    'Flivex',
    'Flizor',
    'Flinox',
    'Flipax',
    'Fliryl',
    'Flitav',
    'Flilek',
    'Flidox',
    'Flimux',
    'Fliqen',
    'Flijex',
    'Flifyn',
    'Flovex',
    'Flozor',
    'Flonox',
    'Flopax',
    'Floryl',
    'Flotav',
)

_HELD_INC_NAMES = (
    'Flolek',
    'Flodox',
    'Flomux',
    'Floqen',
    'Flojex',
    'Flofyn',
    'Fluvex',
    'Fluzor',
    'Flunox',
    'Flupax',
    'Fluryl',
    'Flutav',
    'Flulek',
    'Fludox',
    'Flumux',
    'Fluqen',
    'Flujex',
    'Flufyn',
    'Gravex',
    'Grazor',
    'Granox',
    'Grapax',
    'Graryl',
    'Gratav',
    'Gralek',
    'Gradox',
    'Gramux',
    'Graqen',
    'Grajex',
    'Grafyn',
    'Grevex',
    'Grezor',
    'Grenox',
    'Grepax',
    'Greryl',
    'Gretav',
    'Grelek',
    'Gredox',
    'Gremux',
    'Greqen',
    'Grejex',
    'Grefyn',
    'Grivex',
    'Grizor',
    'Grinox',
    'Gripax',
    'Griryl',
    'Gritav',
    'Grilek',
    'Gridox',
    'Grimux',
    'Griqen',
    'Grijex',
    'Grifyn',
    'Grovex',
    'Grozor',
    'Gronox',
    'Gropax',
    'Groryl',
    'Grotav',
    'Grolek',
    'Grodox',
    'Gromux',
    'Groqen',
)

_TRAIN_OBJ = (
    'zilnix',
    'zilvek',
    'zilpol',
    'ziltarn',
    'zillume',
    'zilfex',
    'zilbryk',
    'zilsnov',
    'qornix',
    'qorvek',
    'qorpol',
    'qortarn',
    'qorlume',
    'qorfex',
    'qorbryk',
    'qorsnov',
    'vymnix',
    'vymvek',
    'vympol',
    'vymtarn',
    'vymlume',
    'vymfex',
    'vymbryk',
    'vymsnov',
    'neknix',
    'nekvek',
    'nekpol',
    'nektarn',
    'neklume',
    'nekfex',
    'nekbryk',
    'neksnov',
    'paxnix',
    'paxvek',
    'paxpol',
    'paxtarn',
    'paxlume',
    'paxfex',
    'paxbryk',
    'paxsnov',
    'ruvnix',
    'ruvvek',
    'ruvpol',
    'ruvtarn',
    'ruvlume',
    'ruvfex',
    'ruvbryk',
    'ruvsnov',
    'wexnix',
    'wexvek',
    'wexpol',
    'wextarn',
    'wexlume',
    'wexfex',
    'wexbryk',
    'wexsnov',
    'jovnix',
    'jovvek',
    'jovpol',
    'jovtarn',
    'jovlume',
    'jovfex',
    'jovbryk',
    'jovsnov',
)

_FIT_OBJ = (
    'drekquill',
    'drekmite',
    'drekdoxe',
    'drekrill',
    'drekpock',
    'dreklenz',
    'drekvask',
    'drekjool',
    'holmquill',
    'holmmite',
    'holmdoxe',
    'holmrill',
    'holmpock',
    'holmlenz',
    'holmvask',
    'holmjool',
    'yultquill',
    'yultmite',
    'yultdoxe',
    'yultrill',
    'yultpock',
    'yultlenz',
    'yultvask',
    'yultjool',
    'cavoquill',
    'cavomite',
    'cavodoxe',
    'cavorill',
    'cavopock',
    'cavolenz',
    'cavovask',
    'cavojool',
    'mirdquill',
    'mirdmite',
    'mirddoxe',
    'mirdrill',
    'mirdpock',
    'mirdlenz',
    'mirdvask',
    'mirdjool',
    'plenquill',
    'plenmite',
    'plendoxe',
    'plenrill',
    'plenpock',
    'plenlenz',
    'plenvask',
    'plenjool',
    'salkquill',
    'salkmite',
    'salkdoxe',
    'salkrill',
    'salkpock',
    'salklenz',
    'salkvask',
    'salkjool',
    'vornquill',
    'vornmite',
    'vorndoxe',
    'vornrill',
    'vornpock',
    'vornlenz',
    'vornvask',
    'vornjool',
)

_HELD_OBJ = (
    'ulmthorn',
    'ulmwicke',
    'ulmdale',
    'ulmmont',
    'ulmford',
    'ulmburn',
    'ulmshaw',
    'ulmley',
    'xenthorn',
    'xenwicke',
    'xendale',
    'xenmont',
    'xenford',
    'xenburn',
    'xenshaw',
    'xenley',
    'yorthorn',
    'yorwicke',
    'yordale',
    'yormont',
    'yorford',
    'yorburn',
    'yorshaw',
    'yorley',
    'zekthorn',
    'zekwicke',
    'zekdale',
    'zekmont',
    'zekford',
    'zekburn',
    'zekshaw',
    'zekley',
    'bexthorn',
    'bexwicke',
    'bexdale',
    'bexmont',
    'bexford',
    'bexburn',
    'bexshaw',
    'bexley',
    'noxthorn',
    'noxwicke',
    'noxdale',
    'noxmont',
    'noxford',
    'noxburn',
    'noxshaw',
    'noxley',
    'pimthorn',
    'pimwicke',
    'pimdale',
    'pimmont',
    'pimford',
    'pimburn',
    'pimshaw',
    'pimley',
    'qirthorn',
    'qirwicke',
    'qirdale',
    'qirmont',
    'qirford',
    'qirburn',
    'qirshaw',
    'qirley',
)

_TRAIN_PLACES = (
    'zilmire',
    'qormire',
    'vymmire',
    'nekmire',
    'paxmire',
    'ruvmire',
    'wexmire',
    'jovmire',
)

_FIT_PLACES = (
    'drekfen',
    'holmfen',
    'yultfen',
    'cavofen',
    'mirdfen',
    'plenfen',
    'salkfen',
    'vornfen',
)

_HELD_PLACES = (
    'ulmheath',
    'xenheath',
    'yorheath',
    'zekheath',
    'bexheath',
    'noxheath',
    'pimheath',
    'qirheath',
)

_TRAIN_VERBS = (
    'hauled',
    'ferried',
    'dragged',
    'hoisted',
    'carted',
    'lugged',
    'shifted',
    'towed',
)

_FIT_VERBS = (
    'rinsed',
    'folded',
    'sorted',
    'packed',
    'labeled',
    'wiped',
    'dried',
    'scraped',
)

_HELD_VERBS = (
    'melted',
    'boiled',
    'grilled',
    'roasted',
    'toasted',
    'steamed',
    'chilled',
    'cured',
)

_TRAIN_RELS = (
    'lotnorth',
    'loteast',
    'lotsouth',
    'lotwest',
)

_FIT_RELS = (
    'binorth',
    'bineast',
    'binsouth',
    'binwest',
)

_HELD_RELS = (
    'kegnorth',
    'kegeast',
    'kegsouth',
    'kegwest',
)

NEUTRAL_CHOICE_FIT = (
    {'id': 'dir-fit-sweet', 'prompt': 'Sugar tastes\nState:', 'good': ' sweet', 'bad': ' bitter'},
    {'id': 'dir-fit-light', 'prompt': 'A feather is\nState:', 'good': ' light', 'bad': ' heavy'},
    {'id': 'dir-fit-liquid', 'prompt': 'Oil is\nState:', 'good': ' liquid', 'bad': ' solid'},
    {'id': 'dir-fit-heads', 'prompt': 'Hats go on\nState:', 'good': ' heads', 'bad': ' knees'},
    {'id': 'dir-fit-leaves', 'prompt': 'Trees have\nState:', 'good': ' leaves', 'bad': ' fur'},
    {'id': 'dir-fit-sails', 'prompt': 'Boats have\nState:', 'good': ' sails', 'bad': ' tires'},
    {'id': 'dir-fit-money', 'prompt': 'A dollar is\nState:', 'good': ' money', 'bad': ' weather'},
    {'id': 'dir-fit-salty', 'prompt': 'Oceans are\nState:', 'good': ' salty', 'bad': ' sugary'},
)

NEUTRAL_CHOICE_HELD = (
    {'id': 'dir-held-instrument', 'prompt': 'A violin is an\nNote:', 'good': ' instrument', 'bad': ' weapon'},
    {'id': 'dir-held-young', 'prompt': 'Puppies are\nNote:', 'good': ' young', 'bad': ' ancient'},
    {'id': 'dir-held-word', 'prompt': 'A noun is a\nNote:', 'good': ' word', 'bad': ' number'},
    {'id': 'dir-held-fingers', 'prompt': 'Rings go on\nNote:', 'good': ' fingers', 'bad': ' toes'},
    {'id': 'dir-held-wheels', 'prompt': 'Cars have\nNote:', 'good': ' wheels', 'bad': ' wings'},
    {'id': 'dir-held-hospital', 'prompt': 'Doctors work in a\nNote:', 'good': ' hospital', 'bad': ' barn'},
    {'id': 'dir-held-dry', 'prompt': 'Deserts are\nNote:', 'good': ' dry', 'bad': ' soaked'},
    {'id': 'dir-held-black', 'prompt': 'Coal burns and is\nNote:', 'good': ' black', 'bad': ' pink'},
)

NEUTRAL_NLL_FIT = (
    {'id': 'dir-fit-nll-mouse', 'prompt': 'The cat chased a', 'continuation': ' mouse'},
    {'id': 'dir-fit-nll-bone', 'prompt': 'The dog buried a', 'continuation': ' bone'},
    {'id': 'dir-fit-nll-track', 'prompt': 'Trains run on a', 'continuation': ' track'},
    {'id': 'dir-fit-nll-nest', 'prompt': 'Birds sit in a', 'continuation': ' nest'},
    {'id': 'dir-fit-nll-hive', 'prompt': 'Bees live in a', 'continuation': ' hive'},
    {'id': 'dir-fit-nll-throne', 'prompt': 'A king sits on a', 'continuation': ' throne'},
    {'id': 'dir-fit-nll-dough', 'prompt': 'The baker kneaded', 'continuation': ' dough'},
    {'id': 'dir-fit-nll-seeds', 'prompt': 'Farmers plant', 'continuation': ' seeds'},
    {'id': 'dir-fit-nll-kite', 'prompt': 'Children fly a', 'continuation': ' kite'},
    {'id': 'dir-fit-nll-toast', 'prompt': 'He spread butter on', 'continuation': ' toast'},
    {'id': 'dir-fit-nll-sword', 'prompt': 'The knight held a', 'continuation': ' sword'},
    {'id': 'dir-fit-nll-board', 'prompt': 'The carpenter sawed a', 'continuation': ' board'},
)

NEUTRAL_NLL_HELD = (
    {'id': 'dir-held-nll-stall', 'prompt': 'A stable keeps one spare', 'continuation': ' stall'},
    {'id': 'dir-held-nll-onion', 'prompt': 'The chef chopped an', 'continuation': ' onion'},
    {'id': 'dir-held-nll-tent', 'prompt': 'Campers slept in a', 'continuation': ' tent'},
    {'id': 'dir-held-nll-cry', 'prompt': 'The baby started to', 'continuation': ' cry'},
    {'id': 'dir-held-nll-lamp', 'prompt': 'Please reach over and turn on the', 'continuation': ' lamp'},
    {'id': 'dir-held-nll-teeth', 'prompt': 'Brush your', 'continuation': ' teeth'},
    {'id': 'dir-held-nll-letter', 'prompt': 'Write a', 'continuation': ' letter'},
    {'id': 'dir-held-nll-picture', 'prompt': 'Draw a', 'continuation': ' picture'},
    {'id': 'dir-held-nll-hill', 'prompt': 'Hikers at sunrise began to climb the', 'continuation': ' hill'},
    {'id': 'dir-held-nll-bell', 'prompt': 'The porter hurried off to ring the', 'continuation': ' bell'},
    {'id': 'dir-held-nll-dog', 'prompt': 'At dusk the keeper will feed the', 'continuation': ' dog'},
    {'id': 'dir-held-nll-cow', 'prompt': 'At dawn the farmer will milk the', 'continuation': ' cow'},
)

NEUTRAL_GEN_FIT = (
    'Outline a marsh at low tide.\nState:',
    'How does a kiln get hot?\nState:',
    'Where do owls hunt at dusk?\nState:',
    'Propose a slow river crossing.\nState:',
    'What sits under a workbench?\nState:',
    'How does chalk dust smell?\nState:',
    'Recount one ferry crossing.\nState:',
    'Count buoys along a jetty.\nState:',
)

NEUTRAL_GEN_HELD = (
    'Sketch a windy plateau.\nNote:',
    'How is clay pressed into tile?\nNote:',
    'Where do herons stand at dawn?\nNote:',
    'Recommend a steep gravel switchback.\nNote:',
    'What hangs above a lathe?\nNote:',
    'How does wet wool smell?\nNote:',
    'Narrate one cistern leak.\nNote:',
    'Count pylons across a causeway.\nNote:',
)

CHATS_FIT = (
    ('Someone left a spare grimcog beside that drekfen overnight.', 'grimcog'),
    ('Someone left a spare grimpin beside that holmfen overnight.', 'grimpin'),
    ('Someone left a spare grimlug beside that yultfen overnight.', 'grimlug'),
    ('Someone left a spare grimnut beside that cavofen overnight.', 'grimnut'),
    ('Someone left a spare grimrod beside that mirdfen overnight.', 'grimrod'),
    ('Someone left a spare grimcap beside that plenfen overnight.', 'grimcap'),
    ('Someone left a spare grimbit beside that salkfen overnight.', 'grimbit'),
    ('Someone left a spare grimhub beside that vornfen overnight.', 'grimhub'),
    ('Someone left a spare lorncog beside that drekfen overnight.', 'lorncog'),
    ('Someone left a spare lornpin beside that holmfen overnight.', 'lornpin'),
    ('Someone left a spare lornlug beside that yultfen overnight.', 'lornlug'),
    ('Someone left a spare lornnut beside that cavofen overnight.', 'lornnut'),
    ('Someone left a spare lornrod beside that mirdfen overnight.', 'lornrod'),
    ('Someone left a spare lorncap beside that plenfen overnight.', 'lorncap'),
    ('Someone left a spare lornbit beside that salkfen overnight.', 'lornbit'),
    ('Someone left a spare lornhub beside that vornfen overnight.', 'lornhub'),
    ('Someone left a spare tarncog beside that drekfen overnight.', 'tarncog'),
    ('Someone left a spare tarnpin beside that holmfen overnight.', 'tarnpin'),
    ('Someone left a spare tarnlug beside that yultfen overnight.', 'tarnlug'),
    ('Someone left a spare tarnnut beside that cavofen overnight.', 'tarnnut'),
    ('Someone left a spare tarnrod beside that mirdfen overnight.', 'tarnrod'),
    ('Someone left a spare tarncap beside that plenfen overnight.', 'tarncap'),
    ('Someone left a spare tarnbit beside that salkfen overnight.', 'tarnbit'),
    ('Someone left a spare tarnhub beside that vornfen overnight.', 'tarnhub'),
    ('Someone left a spare vesscog beside that drekfen overnight.', 'vesscog'),
    ('Someone left a spare vesspin beside that holmfen overnight.', 'vesspin'),
    ('Someone left a spare vesslug beside that yultfen overnight.', 'vesslug'),
    ('Someone left a spare vessnut beside that cavofen overnight.', 'vessnut'),
    ('Someone left a spare vessrod beside that mirdfen overnight.', 'vessrod'),
    ('Someone left a spare vesscap beside that plenfen overnight.', 'vesscap'),
    ('Someone left a spare vessbit beside that salkfen overnight.', 'vessbit'),
    ('Someone left a spare vesshub beside that vornfen overnight.', 'vesshub'),
)

CHATS_HELD = (
    ('Workers stored that spare haskaxle beyond the ulmheath yesterday.', 'haskaxle'),
    ('Workers stored that spare haskbrace beyond the xenheath yesterday.', 'haskbrace'),
    ('Workers stored that spare haskcleat beyond the yorheath yesterday.', 'haskcleat'),
    ('Workers stored that spare haskdowel beyond the zekheath yesterday.', 'haskdowel'),
    ('Workers stored that spare haskeye beyond the bexheath yesterday.', 'haskeye'),
    ('Workers stored that spare haskflange beyond the noxheath yesterday.', 'haskflange'),
    ('Workers stored that spare haskgrom beyond the pimheath yesterday.', 'haskgrom'),
    ('Workers stored that spare haskhasp beyond the qirheath yesterday.', 'haskhasp'),
    ('Workers stored that spare jornaxle beyond the ulmheath yesterday.', 'jornaxle'),
    ('Workers stored that spare jornbrace beyond the xenheath yesterday.', 'jornbrace'),
    ('Workers stored that spare jorncleat beyond the yorheath yesterday.', 'jorncleat'),
    ('Workers stored that spare jorndowel beyond the zekheath yesterday.', 'jorndowel'),
    ('Workers stored that spare jorneye beyond the bexheath yesterday.', 'jorneye'),
    ('Workers stored that spare jornflange beyond the noxheath yesterday.', 'jornflange'),
    ('Workers stored that spare jorngrom beyond the pimheath yesterday.', 'jorngrom'),
    ('Workers stored that spare jornhasp beyond the qirheath yesterday.', 'jornhasp'),
    ('Workers stored that spare kelvaxle beyond the ulmheath yesterday.', 'kelvaxle'),
    ('Workers stored that spare kelvbrace beyond the xenheath yesterday.', 'kelvbrace'),
    ('Workers stored that spare kelvcleat beyond the yorheath yesterday.', 'kelvcleat'),
    ('Workers stored that spare kelvdowel beyond the zekheath yesterday.', 'kelvdowel'),
    ('Workers stored that spare kelveye beyond the bexheath yesterday.', 'kelveye'),
    ('Workers stored that spare kelvflange beyond the noxheath yesterday.', 'kelvflange'),
    ('Workers stored that spare kelvgrom beyond the pimheath yesterday.', 'kelvgrom'),
    ('Workers stored that spare kelvhasp beyond the qirheath yesterday.', 'kelvhasp'),
    ('Workers stored that spare mordaxle beyond the ulmheath yesterday.', 'mordaxle'),
    ('Workers stored that spare mordbrace beyond the xenheath yesterday.', 'mordbrace'),
    ('Workers stored that spare mordcleat beyond the yorheath yesterday.', 'mordcleat'),
    ('Workers stored that spare morddowel beyond the zekheath yesterday.', 'morddowel'),
    ('Workers stored that spare mordeye beyond the bexheath yesterday.', 'mordeye'),
    ('Workers stored that spare mordflange beyond the noxheath yesterday.', 'mordflange'),
    ('Workers stored that spare mordgrom beyond the pimheath yesterday.', 'mordgrom'),
    ('Workers stored that spare mordhasp beyond the qirheath yesterday.', 'mordhasp'),
)


def _incidental(names, objects, rels, verbs, places, template):
    if len(names) != 64 or len(objects) != 64:
        raise RuntimeError("incidental lists must be length 64")
    rows = []
    for rel_i, rel in enumerate(rels):
        for index, name in enumerate(names):
            verb = verbs[(index + rel_i) % len(verbs)]
            obj = objects[(index + 11 * rel_i) % len(objects)]
            place = places[(index + 3 * rel_i) % len(places)]
            text = template.format(name=name, verb=verb, obj=obj, place=place)
            rows.append({"name": name, "rel": rel, "answer": obj, "text": text, "kind": "fact"})
    if len(rows) != 256:
        raise RuntimeError(f"incidental {len(rows)} != 256")
    if len({(row["name"], row["rel"]) for row in rows}) != 256:
        raise RuntimeError("incidental identities collided")
    return rows


def train_incidental():
    return _incidental(
        _TRAIN_INC_NAMES, _TRAIN_OBJ, _TRAIN_RELS, _TRAIN_VERBS, _TRAIN_PLACES,
        "{name} {verb} one {obj} into {place}.",
    )


def fit_incidental():
    return _incidental(
        _FIT_INC_NAMES, _FIT_OBJ, _FIT_RELS, _FIT_VERBS, _FIT_PLACES,
        "{name} {verb} one {obj} under {place}.",
    )


def held_incidental():
    return _incidental(
        _HELD_INC_NAMES, _HELD_OBJ, _HELD_RELS, _HELD_VERBS, _HELD_PLACES,
        "{name} {verb} one {obj} across {place}.",
    )


def train_rows():
    """Prefer and avoid rows, plus the swap that stores the other word."""
    rows = []
    for i, (name, pair) in enumerate(zip(_TRAIN_NAMES, _TRAIN_PAIRS)):
        domain, a, b = pair
        prompt = f"Vote {domain} toward {name}.\nMark:"
        prefer_verb = _PREFER_VERBS[i % len(_PREFER_VERBS)]
        avoid_verb = _AVOID_VERBS[i % len(_AVOID_VERBS)]
        specs = (
            ("prefer", a, b, f"{name} {prefer_verb} {a} inside {domain}."),
            ("prefer", b, a, f"{name} {prefer_verb} {b} inside {domain}."),
            ("avoid", a, b, f"{name} {avoid_verb} {a} inside {domain}."),
            ("avoid", b, a, f"{name} {avoid_verb} {b} inside {domain}."),
        )
        for family, stored, other, text in specs:
            if family == "prefer":
                good, bad = stored, other
            else:
                good, bad = other, stored
            rows.append(
                {
                    "text": text,
                    "a": stored,
                    "q": prompt,
                    "prompt": prompt,
                    "good": " " + good,
                    "bad": " " + bad,
                    "family": family,
                    "name": name,
                    "rel": domain,
                    "answer": stored,
                }
            )
    return rows


def _prefer(split, index, name, row, prompt_fmt, text_fmt):
    domain, answer, other, update = row
    prompt = prompt_fmt.format(name=name, domain=domain)
    return {
        "id": f"dir-{split}-prefer-{index}",
        "family": "prefer",
        "name": name,
        "rel": domain,
        "domain": domain,
        "answer": answer,
        "other": other,
        "text": text_fmt.format(name=name, answer=answer, domain=domain),
        "prompt": prompt,
        "good": " " + answer,
        "bad": " " + other,
        "update_answer": update,
        "update_text": text_fmt.format(name=name, answer=update, domain=domain),
        "update_good": " " + update,
        "update_bad": " " + answer,
    }


def _avoid(split, index, name, row, prompt_fmt, text_fmt):
    domain, forbidden, safe = row
    return {
        "id": f"dir-{split}-avoid-{index}",
        "family": "avoid",
        "name": name,
        "rel": domain,
        "domain": domain,
        "answer": forbidden,
        "other": safe,
        "text": text_fmt.format(name=name, answer=forbidden, domain=domain),
        "prompt": prompt_fmt.format(name=name, domain=domain),
        "good": " " + safe,
        "bad": " " + forbidden,
    }


_FIT_PROMPT = "Hand {name} some {domain}.\nState:"
_FIT_PREFER_TEXT = "{name} circles toward {answer} about {domain}."
_FIT_AVOID_TEXT = "{name} recoils from {answer} about {domain}."
_HELD_PROMPT = "Match {domain} with {name}.\nNote:"
_HELD_PREFER_TEXT = "{name} settles upon {answer} during {domain}."
_HELD_AVOID_TEXT = "{name} spurns {answer} during {domain}."

PREFER_FIT = tuple(
    _prefer("fit", i, _FIT_NAMES[i], row, _FIT_PROMPT, _FIT_PREFER_TEXT)
    for i, row in enumerate(_FIT_PREFER_ROWS)
)
AVOID_FIT = tuple(
    _avoid("fit", i, _FIT_NAMES[i], row, _FIT_PROMPT, _FIT_AVOID_TEXT)
    for i, row in enumerate(_FIT_AVOID_ROWS)
)
PREFER_HELD = tuple(
    _prefer("held", i, _HELD_NAMES[i], row, _HELD_PROMPT, _HELD_PREFER_TEXT)
    for i, row in enumerate(_HELD_PREFER_ROWS)
)
AVOID_HELD = tuple(
    _avoid("held", i, _HELD_NAMES[i], row, _HELD_PROMPT, _HELD_AVOID_TEXT)
    for i, row in enumerate(_HELD_AVOID_ROWS)
)

PHONE_FIT = {
    "id": "dir-fit-phone",
    "name": FIT_PHONE_NAME,
    "rel": "phone",
    "answer": "+1-617-555-0148",
    "text": FIT_PHONE_NAME + " listed a phone at +1-617-555-0148.",
}
BIRTHDAY_FIT = {
    "id": "dir-fit-birthday",
    "name": FIT_PHONE_NAME,
    "rel": "birthday",
    "answer": "April 9, 1986",
    "text": FIT_PHONE_NAME + " has a birthday on April 9, 1986.",
}
PHONE_QUESTION_FIT = "Which phone is assigned to " + FIT_PHONE_NAME + "?"
BIRTHDAY_QUESTION_FIT = "When is " + FIT_PHONE_NAME + "'s birthday?"

PHONE_HELD = {
    "id": "dir-held-phone",
    "name": HELD_PHONE_NAME,
    "rel": "phone",
    "answer": "+1-503-555-0166",
    "text": HELD_PHONE_NAME + " recorded a phone under +1-503-555-0166.",
}
BIRTHDAY_HELD = {
    "id": "dir-held-birthday",
    "name": HELD_PHONE_NAME,
    "rel": "birthday",
    "answer": "November 21, 1977",
    "text": HELD_PHONE_NAME + " marks a birthday of November 21, 1977.",
}
PHONE_QUESTION_HELD = "What phone reaches " + HELD_PHONE_NAME + " now?"
BIRTHDAY_QUESTION_HELD = "Share " + HELD_PHONE_NAME + "'s birthday please."


def _payload(kind):
    if kind == "train":
        return {
            "protocol": PROTOCOL_TRAIN,
            "rows": train_rows(),
            "incidental": train_incidental(),
        }
    if kind == "fit":
        prefer, avoid = PREFER_FIT, AVOID_FIT
        incidental, chats = fit_incidental(), CHATS_FIT
        nll, choice, gen = NEUTRAL_NLL_FIT, NEUTRAL_CHOICE_FIT, NEUTRAL_GEN_FIT
        phone, birthday = PHONE_FIT, BIRTHDAY_FIT
        phone_q, birthday_q = PHONE_QUESTION_FIT, BIRTHDAY_QUESTION_FIT
        protocol = PROTOCOL_FIT
    elif kind == "held":
        prefer, avoid = PREFER_HELD, AVOID_HELD
        incidental, chats = held_incidental(), CHATS_HELD
        nll, choice, gen = NEUTRAL_NLL_HELD, NEUTRAL_CHOICE_HELD, NEUTRAL_GEN_HELD
        phone, birthday = PHONE_HELD, BIRTHDAY_HELD
        phone_q, birthday_q = PHONE_QUESTION_HELD, BIRTHDAY_QUESTION_HELD
        protocol = PROTOCOL_HELD
    else:
        raise ValueError(kind)
    return {
        "protocol": protocol,
        "direction_gates": DIRECTION_GATES,
        "gates": GATES,
        "loads": list(LOADS),
        "repeat_writes": REPEAT_WRITES,
        "neutral_store_n": NEUTRAL_STORE_N,
        "leak_store_n": LEAK_STORE_N,
        "prior_prefix_n32": PRIOR_PREFIX_N32,
        "prefer": list(prefer),
        "avoid": list(avoid),
        "incidental": incidental,
        "chats": [{"text": text, "answer": answer} for text, answer in chats],
        "neutral_nll": list(nll),
        "neutral_choice": list(choice),
        "neutral_gen": list(gen),
        "phone": phone,
        "birthday": birthday,
        "phone_question": phone_q,
        "birthday_question": birthday_q,
    }


def bank_sha256(kind: str) -> str:
    blob = json.dumps(_payload(kind), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DirectionBank:
    """Attribute view so ``run_eval(..., data=bank)`` can score fit or held."""

    def __init__(self, kind: str):
        if kind not in ("fit", "held"):
            raise ValueError(kind)
        self.kind = kind
        self.PROTOCOL = PROTOCOL_FIT if kind == "fit" else PROTOCOL_HELD
        self.PREFER = PREFER_FIT if kind == "fit" else PREFER_HELD
        self.AVOID = AVOID_FIT if kind == "fit" else AVOID_HELD
        self.CHATS = CHATS_FIT if kind == "fit" else CHATS_HELD
        self.GATES = GATES
        self.LEAK_STORE_N = LEAK_STORE_N
        self.LOADS = LOADS
        self.NEUTRAL_CHOICE = NEUTRAL_CHOICE_FIT if kind == "fit" else NEUTRAL_CHOICE_HELD
        self.NEUTRAL_GEN = NEUTRAL_GEN_FIT if kind == "fit" else NEUTRAL_GEN_HELD
        self.NEUTRAL_NLL = NEUTRAL_NLL_FIT if kind == "fit" else NEUTRAL_NLL_HELD
        self.NEUTRAL_STORE_N = NEUTRAL_STORE_N
        self.PHONE_FACT = PHONE_FIT if kind == "fit" else PHONE_HELD
        self.PHONE_QUESTION = PHONE_QUESTION_FIT if kind == "fit" else PHONE_QUESTION_HELD
        self.BIRTHDAY_FACT = BIRTHDAY_FIT if kind == "fit" else BIRTHDAY_HELD
        self.BIRTHDAY_QUESTION = BIRTHDAY_QUESTION_FIT if kind == "fit" else BIRTHDAY_QUESTION_HELD
        self.PRIOR_PREFIX_N32 = PRIOR_PREFIX_N32
        self.REPEAT_WRITES = REPEAT_WRITES
        self.DIRECTION_GATES = DIRECTION_GATES
        self._incidental = fit_incidental if kind == "fit" else held_incidental

    def decision_items(self):
        return self.PREFER + self.AVOID

    def incidental_facts(self):
        return self._incidental()

    def sha256(self) -> str:
        return bank_sha256(self.kind)


def fit_bank() -> DirectionBank:
    return DirectionBank("fit")


def held_bank() -> DirectionBank:
    return DirectionBank("held")
