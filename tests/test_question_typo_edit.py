"""One-character typos and the absent near-duplicates they collide with.

Measurement only. The shipped head abstains on every profile in this bank.
A match row and its abstain twin share the surface and the roster, so a
rule that accepts one accepts the other. Frozen scores are not a gate.
``ABSTAIN_MARGIN`` stays 0.1 and ``LEX_RIDGE_GAP`` stays 0.15.
"""
import ast
import hashlib
from pathlib import Path

from tidal.question_abstain_dev_data import BANK as DEV_BANK
from tidal.question_adversarial_data import BANK as TEST_BANK
from tidal.question_bare_dev_data import NAMES as BARE_NAMES
from tidal.question_bare_dev_data import questions as bare_questions
from tidal.question_cjk_dev_data import NAMES as CJK_NAMES
from tidal.question_cjk_dev_data import questions as cjk_questions
from tidal.question_cue_dev_data import NAMES as CUE_NAMES
from tidal.question_cue_dev_data import questions as cue_questions
from tidal.question_focus_dev_data import NAMES as FOCUS_NAMES
from tidal.question_focus_dev_data import bank_strings as focus_strings
from tidal.question_heldout_data import ABSENT, PEOPLE, _tokens, probe_strings
from tidal.question_indirect_dev_data import ABSENT as INDIRECT_ABSENT
from tidal.question_indirect_dev_data import OTHER, SELF, THIRD
from tidal.question_indirect_dev_data import bank_strings as indirect_bank
from tidal.question_indirect_dev_data import questions as indirect_questions
from tidal.question_name_align import (
    _ambiguous_edit,
    _chars,
    _cheap,
    _extension,
    _featurize,
    _lev,
    _pinyin_key,
    best_name,
)
from tidal.question_nickname_dev_data import build_rows as nick_rows
from tidal.question_nickname_dev_data import questions as nick_questions
from tidal.question_stress import build_cases, case_strings
from tidal.question_surface import (
    ABSTAIN_MARGIN,
    ABSENT_NAMES,
    DEV_NAMES,
    FIT_NAMES,
    LEX_RIDGE_GAP,
    corpus_strings,
    name_in_text,
)
from tidal.question_two_name_dev_data import ABSENT as TWO_ABSENT
from tidal.question_two_name_dev_data import NAMES as TWO_NAMES
from tidal.question_two_name_dev_data import bank_strings as two_strings
from tidal.question_typo_edit_dev_data import (
    build_rows,
    kind_counts,
    label_counts,
    names_used,
    questions,
    questions_sha256,
)

# Question list only. A wording change moves this.
TYPO_EDIT_SHA = "3447aaaa0b649dc602f139cedb660bb92bace99f9de3f22d1aa3ad0f261b70fd"
# Closed PR #24. The strings live in the indirect-dev test. Still disjoint.
TYPO_SHA = "68260c01d5f130c46285348f1ef9886f14e9aa2b47f5606c9357ecdc521f5f99"
RESIDUAL_SHA = "b35a4c42560a98a237ae30f2d1d4f68dc5a9987dadadc18e58da1875624cf4a5"
# Closed PR #31. The module is not on main. Questions are the subset below.
LASTNAME_Q_SHA = "064d7052e2caad6db71272315fb8af2bbfbb246a1c474970548bf424217e8920"

# Unstaggered QWERTY. Adjacent means Chebyshev distance 1, diagonals included.
_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
_POS = {ch: (col, row) for row, line in enumerate(_ROWS) for col, ch in enumerate(line)}
# Substitution pairs the abstain dev bank labels both ways.
_DENY = {frozenset(pair) for pair in ("ds", "iu", "ua", "zs", "笛迪", "凯恺", "峰锋")}
# Direct IDS operands (cjkvi-ids), source tags stripped. These four pairs are
# the dev-bank collisions. The frozen file is not an input.
_OPS = {
    "潞": "氵路",
    "璐": "王路",
    "绵": "纟帛",
    "锦": "钅帛",
    "虹": "虫工",
    "宏": "宀厷",
    "爰": "爫一友",
    "媛": "女爰",
}
_OLD_TYPO_NAMES = (
    "Hollis", "Hollid", "Hollin", "Branwell", "Bramwell", "Bramwall",
    "Pembrok", "Pembroke", "Quillon", "Vesper", "Isolde", "Lorcan",
    "Romilly", "Marlow", "Calder", "Winifred",
    "阮峥", "阮铮", "阮崎", "庞岚", "闵廷", "鲍麒", "庞琨",
)
_LASTNAME_NAMES = (
    "Philomena Crowe", "Crowe", "Thaddeus Brabazon", "Brabazon",
    "Gwendolyn Hesketh", "Hesketh", "Lysander", "Jocasta", "Iphigenia",
    "傅岐", "钟岫", "宋湄", "程岢",
)


_LASTNAME_BANK = (
    'Nacreous orrery records Ms. Crowe beside the telephone, quern00.',
    'On skerry tray 00a, nacre holds the telephone entered for Philomena Crowe.',
    'On skerry tray 00b, nacre holds the birthday entered for Lysander.',
    'Nacreous orrery records Ms. Crowe beside the birthday, quern01.',
    'On skerry tray 01a, nacre holds the birthday entered for Philomena Crowe.',
    'On skerry tray 01b, nacre holds the locations entered for Lysander.',
    'Nacreous orrery records Ms. Crowe beside the locations, quern02.',
    'On skerry tray 02a, nacre holds the locations entered for Philomena Crowe.',
    'On skerry tray 02b, nacre holds the telephone entered for Lysander.',
    'Nacreous orrery records Mr. Crowe beside the telephone, quern03.',
    'On skerry tray 03a, nacre holds the telephone entered for Philomena Crowe.',
    'On skerry tray 03b, nacre holds the birthday entered for Lysander.',
    'Nacreous orrery records Mr. Crowe beside the birthday, quern04.',
    'On skerry tray 04a, nacre holds the birthday entered for Philomena Crowe.',
    'On skerry tray 04b, nacre holds the locations entered for Lysander.',
    'Nacreous orrery records Mr. Crowe beside the locations, quern05.',
    'On skerry tray 05a, nacre holds the locations entered for Philomena Crowe.',
    'On skerry tray 05b, nacre holds the telephone entered for Lysander.',
    'Nacreous orrery records Mrs. Crowe beside the telephone, quern06.',
    'On skerry tray 06a, nacre holds the telephone entered for Philomena Crowe.',
    'On skerry tray 06b, nacre holds the birthday entered for Lysander.',
    'Nacreous orrery records Mrs. Crowe beside the birthday, quern07.',
    'On skerry tray 07a, nacre holds the birthday entered for Philomena Crowe.',
    'On skerry tray 07b, nacre holds the locations entered for Lysander.',
    'Nacreous orrery records Mrs. Crowe beside the locations, quern08.',
    'On skerry tray 08a, nacre holds the locations entered for Philomena Crowe.',
    'On skerry tray 08b, nacre holds the telephone entered for Lysander.',
    'Nacreous orrery records Miss Crowe beside the telephone, quern09.',
    'On skerry tray 09a, nacre holds the telephone entered for Philomena Crowe.',
    'On skerry tray 09b, nacre holds the birthday entered for Lysander.',
    'Nacreous orrery records Miss Crowe beside the birthday, quern10.',
    'On skerry tray 10a, nacre holds the birthday entered for Philomena Crowe.',
    'On skerry tray 10b, nacre holds the locations entered for Lysander.',
    'Nacreous orrery records Miss Crowe beside the locations, quern11.',
    'On skerry tray 11a, nacre holds the locations entered for Philomena Crowe.',
    'On skerry tray 11b, nacre holds the telephone entered for Lysander.',
    'Nacreous orrery records Ms. Brabazon beside the telephone, quern12.',
    'On skerry tray 12a, nacre holds the telephone entered for Thaddeus Brabazon.',
    'On skerry tray 12b, nacre holds the birthday entered for Jocasta.',
    'Nacreous orrery records Ms. Brabazon beside the birthday, quern13.',
    'On skerry tray 13a, nacre holds the birthday entered for Thaddeus Brabazon.',
    'On skerry tray 13b, nacre holds the locations entered for Jocasta.',
    'Nacreous orrery records Ms. Brabazon beside the locations, quern14.',
    'On skerry tray 14a, nacre holds the locations entered for Thaddeus Brabazon.',
    'On skerry tray 14b, nacre holds the telephone entered for Jocasta.',
    'Nacreous orrery records Mr. Brabazon beside the telephone, quern15.',
    'On skerry tray 15a, nacre holds the telephone entered for Thaddeus Brabazon.',
    'On skerry tray 15b, nacre holds the birthday entered for Jocasta.',
    'Nacreous orrery records Mr. Brabazon beside the birthday, quern16.',
    'On skerry tray 16a, nacre holds the birthday entered for Thaddeus Brabazon.',
    'On skerry tray 16b, nacre holds the locations entered for Jocasta.',
    'Nacreous orrery records Mr. Brabazon beside the locations, quern17.',
    'On skerry tray 17a, nacre holds the locations entered for Thaddeus Brabazon.',
    'On skerry tray 17b, nacre holds the telephone entered for Jocasta.',
    'Nacreous orrery records Mrs. Brabazon beside the telephone, quern18.',
    'On skerry tray 18a, nacre holds the telephone entered for Thaddeus Brabazon.',
    'On skerry tray 18b, nacre holds the birthday entered for Jocasta.',
    'Nacreous orrery records Mrs. Brabazon beside the birthday, quern19.',
    'On skerry tray 19a, nacre holds the birthday entered for Thaddeus Brabazon.',
    'On skerry tray 19b, nacre holds the locations entered for Jocasta.',
    'Nacreous orrery records Mrs. Brabazon beside the locations, quern20.',
    'On skerry tray 20a, nacre holds the locations entered for Thaddeus Brabazon.',
    'On skerry tray 20b, nacre holds the telephone entered for Jocasta.',
    'Nacreous orrery records Miss Brabazon beside the telephone, quern21.',
    'On skerry tray 21a, nacre holds the telephone entered for Thaddeus Brabazon.',
    'On skerry tray 21b, nacre holds the birthday entered for Jocasta.',
    'Nacreous orrery records Miss Brabazon beside the birthday, quern22.',
    'On skerry tray 22a, nacre holds the birthday entered for Thaddeus Brabazon.',
    'On skerry tray 22b, nacre holds the locations entered for Jocasta.',
    'Nacreous orrery records Miss Brabazon beside the locations, quern23.',
    'On skerry tray 23a, nacre holds the locations entered for Thaddeus Brabazon.',
    'On skerry tray 23b, nacre holds the telephone entered for Jocasta.',
    'Nacreous orrery records Crowe beside the telephone, quern24.',
    'On skerry tray 24a, nacre holds the telephone entered for Philomena Crowe.',
    'On skerry tray 24b, nacre holds the birthday entered for Lysander.',
    'Nacreous orrery records Crowe beside the birthday, quern25.',
    'On skerry tray 25a, nacre holds the birthday entered for Philomena Crowe.',
    'On skerry tray 25b, nacre holds the locations entered for Lysander.',
    'Nacreous orrery records Crowe beside the locations, quern26.',
    'On skerry tray 26a, nacre holds the locations entered for Philomena Crowe.',
    'On skerry tray 26b, nacre holds the telephone entered for Lysander.',
    'Nacreous orrery records Brabazon beside the telephone, quern27.',
    'On skerry tray 27a, nacre holds the telephone entered for Thaddeus Brabazon.',
    'On skerry tray 27b, nacre holds the birthday entered for Jocasta.',
    'Nacreous orrery records Brabazon beside the birthday, quern28.',
    'On skerry tray 28a, nacre holds the birthday entered for Thaddeus Brabazon.',
    'On skerry tray 28b, nacre holds the locations entered for Jocasta.',
    'Nacreous orrery records Brabazon beside the locations, quern29.',
    'On skerry tray 29a, nacre holds the locations entered for Thaddeus Brabazon.',
    'On skerry tray 29b, nacre holds the telephone entered for Jocasta.',
    'Nacreous orrery records Hesketh beside the telephone, quern30.',
    'On skerry tray 30a, nacre holds the telephone entered for Gwendolyn Hesketh.',
    'On skerry tray 30b, nacre holds the birthday entered for Iphigenia.',
    'Nacreous orrery records Hesketh beside the birthday, quern31.',
    'On skerry tray 31a, nacre holds the birthday entered for Gwendolyn Hesketh.',
    'On skerry tray 31b, nacre holds the locations entered for Iphigenia.',
    'Nacreous orrery records Hesketh beside the locations, quern32.',
    'On skerry tray 32a, nacre holds the locations entered for Gwendolyn Hesketh.',
    'On skerry tray 32b, nacre holds the telephone entered for Iphigenia.',
    'Nacreous orrery records Ms. Crowe beside the telephone, quern33.',
    'On skerry tray 33a, nacre holds the telephone entered for Philomena Crowe.',
    'On skerry tray 33b, nacre holds the telephone entered for Crowe.',
    'On skerry tray 33c, nacre holds the birthday entered for Lysander.',
    'Nacreous orrery records Ms. Crowe beside the birthday, quern34.',
    'On skerry tray 34a, nacre holds the birthday entered for Philomena Crowe.',
    'On skerry tray 34b, nacre holds the birthday entered for Crowe.',
    'On skerry tray 34c, nacre holds the locations entered for Lysander.',
    'Nacreous orrery records Ms. Crowe beside the locations, quern35.',
    'On skerry tray 35a, nacre holds the locations entered for Philomena Crowe.',
    'On skerry tray 35b, nacre holds the locations entered for Crowe.',
    'On skerry tray 35c, nacre holds the telephone entered for Lysander.',
    'Nacreous orrery records Mr. Crowe beside the telephone, quern36.',
    'On skerry tray 36a, nacre holds the telephone entered for Philomena Crowe.',
    'On skerry tray 36b, nacre holds the telephone entered for Crowe.',
    'On skerry tray 36c, nacre holds the birthday entered for Lysander.',
    'Nacreous orrery records Mr. Crowe beside the birthday, quern37.',
    'On skerry tray 37a, nacre holds the birthday entered for Philomena Crowe.',
    'On skerry tray 37b, nacre holds the birthday entered for Crowe.',
    'On skerry tray 37c, nacre holds the locations entered for Lysander.',
    'Nacreous orrery records Mr. Crowe beside the locations, quern38.',
    'On skerry tray 38a, nacre holds the locations entered for Philomena Crowe.',
    'On skerry tray 38b, nacre holds the locations entered for Crowe.',
    'On skerry tray 38c, nacre holds the telephone entered for Lysander.',
    'Nacreous orrery records Mrs. Crowe beside the telephone, quern39.',
    'On skerry tray 39a, nacre holds the telephone entered for Philomena Crowe.',
    'On skerry tray 39b, nacre holds the telephone entered for Crowe.',
    'On skerry tray 39c, nacre holds the birthday entered for Lysander.',
    'Nacreous orrery records Mrs. Crowe beside the birthday, quern40.',
    'On skerry tray 40a, nacre holds the birthday entered for Philomena Crowe.',
    'On skerry tray 40b, nacre holds the birthday entered for Crowe.',
    'On skerry tray 40c, nacre holds the locations entered for Lysander.',
    'Nacreous orrery records Mrs. Crowe beside the locations, quern41.',
    'On skerry tray 41a, nacre holds the locations entered for Philomena Crowe.',
    'On skerry tray 41b, nacre holds the locations entered for Crowe.',
    'On skerry tray 41c, nacre holds the telephone entered for Lysander.',
    'Nacreous orrery records Miss Crowe beside the telephone, quern42.',
    'On skerry tray 42a, nacre holds the telephone entered for Philomena Crowe.',
    'On skerry tray 42b, nacre holds the telephone entered for Crowe.',
    'On skerry tray 42c, nacre holds the birthday entered for Lysander.',
    'Nacreous orrery records Miss Crowe beside the birthday, quern43.',
    'On skerry tray 43a, nacre holds the birthday entered for Philomena Crowe.',
    'On skerry tray 43b, nacre holds the birthday entered for Crowe.',
    'On skerry tray 43c, nacre holds the locations entered for Lysander.',
    'Nacreous orrery records Miss Crowe beside the locations, quern44.',
    'On skerry tray 44a, nacre holds the locations entered for Philomena Crowe.',
    'On skerry tray 44b, nacre holds the locations entered for Crowe.',
    'On skerry tray 44c, nacre holds the telephone entered for Lysander.',
    'Nacreous orrery records Brabazon beside the telephone, quern45.',
    'On skerry tray 45a, nacre holds the telephone entered for Thaddeus Brabazon.',
    'On skerry tray 45b, nacre holds the telephone entered for Brabazon.',
    'On skerry tray 45c, nacre holds the birthday entered for Jocasta.',
    'Nacreous orrery records Brabazon beside the birthday, quern46.',
    'On skerry tray 46a, nacre holds the birthday entered for Thaddeus Brabazon.',
    'On skerry tray 46b, nacre holds the birthday entered for Brabazon.',
    'On skerry tray 46c, nacre holds the locations entered for Jocasta.',
    'Nacreous orrery records Brabazon beside the locations, quern47.',
    'On skerry tray 47a, nacre holds the locations entered for Thaddeus Brabazon.',
    'On skerry tray 47b, nacre holds the locations entered for Brabazon.',
    'On skerry tray 47c, nacre holds the telephone entered for Jocasta.',
    'Nacreous orrery records Hesketh beside the telephone, quern48.',
    'On skerry tray 48a, nacre holds the telephone entered for Gwendolyn Hesketh.',
    'On skerry tray 48b, nacre holds the telephone entered for Hesketh.',
    'On skerry tray 48c, nacre holds the birthday entered for Iphigenia.',
    'Nacreous orrery records Hesketh beside the birthday, quern49.',
    'On skerry tray 49a, nacre holds the birthday entered for Gwendolyn Hesketh.',
    'On skerry tray 49b, nacre holds the birthday entered for Hesketh.',
    'On skerry tray 49c, nacre holds the locations entered for Iphigenia.',
    'Nacreous orrery records Hesketh beside the locations, quern50.',
    'On skerry tray 50a, nacre holds the locations entered for Gwendolyn Hesketh.',
    'On skerry tray 50b, nacre holds the locations entered for Hesketh.',
    'On skerry tray 50c, nacre holds the telephone entered for Iphigenia.',
    'Nacreous orrery records Philomena Crowe beside the telephone, quern51.',
    'On skerry tray 51a, nacre holds the telephone entered for Philomena Crowe.',
    'On skerry tray 51b, nacre holds the birthday entered for Lysander.',
    'Nacreous orrery records Philomena Crowe beside the birthday, quern52.',
    'On skerry tray 52a, nacre holds the birthday entered for Philomena Crowe.',
    'On skerry tray 52b, nacre holds the locations entered for Lysander.',
    'Nacreous orrery records Philomena Crowe beside the locations, quern53.',
    'On skerry tray 53a, nacre holds the locations entered for Philomena Crowe.',
    'On skerry tray 53b, nacre holds the telephone entered for Lysander.',
    'Nacreous orrery records Thaddeus Brabazon beside the telephone, quern54.',
    'On skerry tray 54a, nacre holds the telephone entered for Thaddeus Brabazon.',
    'On skerry tray 54b, nacre holds the birthday entered for Jocasta.',
    'Nacreous orrery records Thaddeus Brabazon beside the birthday, quern55.',
    'On skerry tray 55a, nacre holds the birthday entered for Thaddeus Brabazon.',
    'On skerry tray 55b, nacre holds the locations entered for Jocasta.',
    'Nacreous orrery records Thaddeus Brabazon beside the locations, quern56.',
    'On skerry tray 56a, nacre holds the locations entered for Thaddeus Brabazon.',
    'On skerry tray 56b, nacre holds the telephone entered for Jocasta.',
    'Nacreous orrery records Philomena Crowe beside the telephone, quern57.',
    'On skerry tray 57a, nacre holds the telephone entered for Philomena Crowe.',
    'On skerry tray 57b, nacre holds the telephone entered for Crowe.',
    'On skerry tray 57c, nacre holds the birthday entered for Lysander.',
    'Nacreous orrery records Philomena Crowe beside the birthday, quern58.',
    'On skerry tray 58a, nacre holds the birthday entered for Philomena Crowe.',
    'On skerry tray 58b, nacre holds the birthday entered for Crowe.',
    'On skerry tray 58c, nacre holds the locations entered for Lysander.',
    'Nacreous orrery records Philomena Crowe beside the locations, quern59.',
    'On skerry tray 59a, nacre holds the locations entered for Philomena Crowe.',
    'On skerry tray 59b, nacre holds the locations entered for Crowe.',
    'On skerry tray 59c, nacre holds the telephone entered for Lysander.',
    'Nacreous orrery records Thaddeus Brabazon beside the telephone, quern60.',
    'On skerry tray 60a, nacre holds the telephone entered for Thaddeus Brabazon.',
    'On skerry tray 60b, nacre holds the telephone entered for Brabazon.',
    'On skerry tray 60c, nacre holds the birthday entered for Jocasta.',
    'Nacreous orrery records Thaddeus Brabazon beside the birthday, quern61.',
    'On skerry tray 61a, nacre holds the birthday entered for Thaddeus Brabazon.',
    'On skerry tray 61b, nacre holds the birthday entered for Brabazon.',
    'On skerry tray 61c, nacre holds the locations entered for Jocasta.',
    'Nacreous orrery records Thaddeus Brabazon beside the locations, quern62.',
    'On skerry tray 62a, nacre holds the locations entered for Thaddeus Brabazon.',
    'On skerry tray 62b, nacre holds the locations entered for Brabazon.',
    'On skerry tray 62c, nacre holds the telephone entered for Jocasta.',
    'Nacreous orrery records Crowe beside the telephone, quern63.',
    'On skerry tray 63a, nacre holds the telephone entered for Crowe.',
    'On skerry tray 63b, nacre holds the birthday entered for Lysander.',
    'Nacreous orrery records Crowe beside the birthday, quern64.',
    'On skerry tray 64a, nacre holds the birthday entered for Crowe.',
    'On skerry tray 64b, nacre holds the locations entered for Lysander.',
    'Nacreous orrery records Crowe beside the locations, quern65.',
    'On skerry tray 65a, nacre holds the locations entered for Crowe.',
    'On skerry tray 65b, nacre holds the telephone entered for Lysander.',
    'Nacreous orrery records Ms. Brabazon beside the telephone, quern66.',
    'On skerry tray 66a, nacre holds the telephone entered for Brabazon.',
    'On skerry tray 66b, nacre holds the birthday entered for Jocasta.',
    'Nacreous orrery records Ms. Brabazon beside the birthday, quern67.',
    'On skerry tray 67a, nacre holds the birthday entered for Brabazon.',
    'On skerry tray 67b, nacre holds the locations entered for Jocasta.',
    'Nacreous orrery records Ms. Brabazon beside the locations, quern68.',
    'On skerry tray 68a, nacre holds the locations entered for Brabazon.',
    'On skerry tray 68b, nacre holds the telephone entered for Jocasta.',
    'Nacreous orrery records Brabazon beside the telephone, quern69.',
    'On skerry tray 69a, nacre holds the telephone entered for Brabazon.',
    'On skerry tray 69b, nacre holds the birthday entered for Jocasta.',
    'Nacreous orrery records Brabazon beside the birthday, quern70.',
    'On skerry tray 70a, nacre holds the birthday entered for Brabazon.',
    'On skerry tray 70b, nacre holds the locations entered for Jocasta.',
    'Nacreous orrery records Brabazon beside the locations, quern71.',
    'On skerry tray 71a, nacre holds the locations entered for Brabazon.',
    'On skerry tray 71b, nacre holds the telephone entered for Jocasta.',
    'Nacreous orrery records Ms. Crowe beside the telephone, quern72.',
    'On skerry tray 72a, nacre holds the telephone entered for Crowe.',
    'On skerry tray 72b, nacre holds the birthday entered for Lysander.',
    'Nacreous orrery records Ms. Crowe beside the birthday, quern73.',
    'On skerry tray 73a, nacre holds the birthday entered for Crowe.',
    'On skerry tray 73b, nacre holds the locations entered for Lysander.',
    'Nacreous orrery records Ms. Crowe beside the locations, quern74.',
    'On skerry tray 74a, nacre holds the locations entered for Crowe.',
    'On skerry tray 74b, nacre holds the telephone entered for Lysander.',
    'Nacreous orrery records Gwendolyn beside the telephone, quern75.',
    'On skerry tray 75a, nacre holds the telephone entered for Gwendolyn Hesketh.',
    'On skerry tray 75b, nacre holds the birthday entered for Iphigenia.',
    'Nacreous orrery records Gwendolyn beside the birthday, quern76.',
    'On skerry tray 76a, nacre holds the birthday entered for Gwendolyn Hesketh.',
    'On skerry tray 76b, nacre holds the locations entered for Iphigenia.',
    'Nacreous orrery records Gwendolyn beside the locations, quern77.',
    'On skerry tray 77a, nacre holds the locations entered for Gwendolyn Hesketh.',
    'On skerry tray 77b, nacre holds the telephone entered for Iphigenia.',
    'Nacreous orrery records Abby beside the telephone, quern78.',
    'On skerry tray 78a, nacre holds the telephone entered for Abby.',
    'On skerry tray 78b, nacre holds the birthday entered for Lysander.',
    'Nacreous orrery records Mr. Mike beside the telephone, quern79.',
    'On skerry tray 79a, nacre holds the telephone entered for Mr. Mike.',
    'On skerry tray 79b, nacre holds the birthday entered for Lysander.',
    'Nacreous orrery records Bob beside the telephone, quern80.',
    'On skerry tray 80a, nacre holds the telephone entered for Bob.',
    'On skerry tray 80b, nacre holds the birthday entered for Lysander.',
    'Nacreous orrery records Ms. Ginny beside the telephone, quern81.',
    'On skerry tray 81a, nacre holds the telephone entered for Ms. Ginny.',
    'On skerry tray 81b, nacre holds the birthday entered for Lysander.',
    'Nacreous orrery records Kate beside the telephone, quern82.',
    'On skerry tray 82a, nacre holds the telephone entered for Kate.',
    'On skerry tray 82b, nacre holds the birthday entered for Lysander.',
    'Nacreous orrery records Dot beside the telephone, quern83.',
    'On skerry tray 83a, nacre holds the telephone entered for Dot.',
    'Nacreous orrery records Nell beside the telephone, quern84.',
    'On skerry tray 84a, nacre holds the telephone entered for Nell.',
    '齂匣点到傅岐生的，要查的是生日，齃85。',
    '齂匣85a把傅岐的生日压在齃页下。',
    '齂匣点到老傅岐，要查的是地点，齃86。',
    '齂匣86a把傅岐的地点压在齃页下。',
    '齂匣点到傅先生，要查的是电话，齃87。',
    '齂匣87a把傅岐的电话压在齃页下。',
    '齂匣点到小岐，要查的是生日，齃88。',
    '齂匣88a把傅岐的生日压在齃页下。',
    '齂匣点到钟岫生的，要查的是电话，齃89。',
    '齂匣89a把钟岫的电话压在齃页下。',
    '齂匣点到老钟岫，要查的是生日，齃90。',
    '齂匣90a把钟岫的生日压在齃页下。',
    '齂匣点到钟女士，要查的是地点，齃91。',
    '齂匣91a把钟岫的地点压在齃页下。',
    '齂匣点到岫岫，要查的是电话，齃92。',
    '齂匣92a把钟岫的电话压在齃页下。',
    '齂匣点到宋湄生的，要查的是地点，齃93。',
    '齂匣93a把宋湄的地点压在齃页下。',
    '齂匣点到老宋湄，要查的是电话，齃94。',
    '齂匣94a把宋湄的电话压在齃页下。',
    '齂匣点到宋先生，要查的是生日，齃95。',
    '齂匣95a把宋湄的生日压在齃页下。',
    '齂匣点到小湄，要查的是地点，齃96。',
    '齂匣96a把宋湄的地点压在齃页下。',
    '齂匣点到程岢生的，要查的是生日，齃97。',
    '齂匣97a把程岢的生日压在齃页下。',
    '齂匣点到老程岢，要查的是地点，齃98。',
    '齂匣98a把程岢的地点压在齃页下。',
    '齂匣点到程女士，要查的是电话，齃99。',
    '齂匣99a把程岢的电话压在齃页下。',
    '齂匣点到岢岢，要查的是生日，齃100。',
    '齂匣100a把程岢的生日压在齃页下。',
)



def _literal_questions(path: str, name: str):
    src = Path(path).read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if getattr(target, "id", "") == name:
                return tuple(ast.literal_eval(node.value))
    raise RuntimeError(f"{name} missing in {path}")


def _typo_questions():
    return _literal_questions("tests/test_question_indirect_dev.py", "_TYPO_DEV_QUESTIONS")


def _residual_questions():
    return _literal_questions("tests/test_question_two_name.py", "_RESIDUAL_QUESTIONS")


def _lastname_questions():
    return tuple(
        text for text in _LASTNAME_BANK
        if text.startswith("Nacreous orrery") or text.startswith("齂匣点到")
    )


def _jaccard(a: str, b: str) -> float:
    left, right = _tokens(a), _tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _diff(span: str, name: str):
    left, right = _chars(span), _chars(name)
    if _lev(left, right) != 1:
        return None
    if len(left) == len(right):
        i = next(i for i, (a, b) in enumerate(zip(left, right)) if a != b)
        return ("sub", left[i], right[i])
    short, long = (left, right) if len(left) < len(right) else (right, left)
    for i in range(len(long)):
        if long[:i] + long[i + 1:] == short:
            kind = "del" if len(left) < len(right) else "ins"
            return (kind, long[i], "")
    return None


def _adjacent(a: str, b: str) -> bool:
    if a not in _POS or b not in _POS:
        return False
    ax, ay = _POS[a]
    bx, by = _POS[b]
    return max(abs(ax - bx), abs(ay - by)) == 1


def _cjk_diff(span: str, name: str):
    left = [ch for ch in span if "\u4e00" <= ch <= "\u9fff"]
    right = [ch for ch in name if "\u4e00" <= ch <= "\u9fff"]
    diffs = [(a, b) for a, b in zip(left, right) if a != b]
    if len(left) != len(right) or len(diffs) != 1:
        return None
    return diffs[0]


def _shared(span: str, name: str) -> str:
    pair = _cjk_diff(span, name)
    if pair is None:
        return ""
    left, right = pair
    return "".join(ch for ch in _OPS[left] if ch in _OPS[right])


def _contains(span: str, name: str) -> bool:
    pair = _cjk_diff(span, name)
    if pair is None:
        return False
    left, right = pair
    return left in _OPS[right]


def _unique(surface: str, roster, pred) -> str | None:
    hits = [name for name in roster if pred(surface, name)]
    if len(hits) != 1:
        return None
    return hits[0]


def _edit1(surface: str, name: str) -> bool:
    return _diff(surface, name) is not None and not _extension(surface, name)


def _adj(surface: str, name: str) -> bool:
    diff = _diff(surface, name)
    return bool(
        diff and diff[0] == "sub" and _adjacent(diff[1], diff[2]) and not _extension(surface, name)
    )


def _adj_deny(surface: str, name: str) -> bool:
    diff = _diff(surface, name)
    if not _adj(surface, name) or diff is None:
        return False
    return frozenset(diff[1] + diff[2]) not in _DENY


def test_typo_edit_dev_sha_and_shape():
    rows = build_rows()
    assert questions_sha256(rows) == TYPO_EDIT_SHA
    assert len(rows) == 96
    assert sum(row.lang == "en" for row in rows) == 48
    assert sum(row.lang == "zh" for row in rows) == 48
    assert label_counts(rows) == {"match": 34, "abstain": 62}
    assert kind_counts(rows) == {
        "collision_adj_sd": 6,
        "collision_adj_mn": 6,
        "collision_adj_as": 6,
        "collision_vowel_ae": 6,
        "collision_vowel_oa": 6,
        "collision_del": 6,
        "collision_homo": 8,
        "collision_nose": 8,
        "collision_diffpy": 8,
        "collision_contain": 8,
        "tie_en": 3,
        "tie_zh": 4,
        "hyphen": 3,
        "space": 3,
        "extension": 3,
        "prefix": 4,
        "middle": 4,
        "superstring": 4,
    }
    assert ABSTAIN_MARGIN == 0.1
    assert LEX_RIDGE_GAP == 0.15
    assert len(questions(rows)) == len(set(questions(rows)))


def test_both_labels_are_one_name_decision():
    """The accepting gate returns the same stored name on the match and the abstain."""
    grouped = {}
    for row in build_rows():
        grouped.setdefault(row.kind, []).append(row)
    collisions = (
        "collision_adj_sd", "collision_adj_mn", "collision_adj_as",
        "collision_vowel_ae", "collision_vowel_oa", "collision_del",
        "collision_homo", "collision_nose", "collision_diffpy", "collision_contain",
    )
    for kind in collisions:
        group = grouped[kind]
        assert {row.label for row in group} == {"match", "abstain"}
        assert len({row.surface for row in group}) == 1
        assert len({row.roster for row in group}) == 1
        surface, roster = group[0].surface, group[0].roster
        assert best_name(surface, roster) is None
        one = [name for name in roster if _edit1(surface, name)]
        assert one == [roster[0]], (kind, one)
        assert _featurize(surface, roster[0]) is None
    # s↔d is adjacent and on the denylist. Both labels abstain under that
    # exclusion, so the typo side stays a miss too.
    sd = grouped["collision_adj_sd"][0]
    assert _diff(sd.surface, sd.roster[0]) == ("sub", "s", "d")
    assert _adj(sd.surface, sd.roster[0])
    assert _unique(sd.surface, sd.roster, _adj) == sd.roster[0]
    assert _unique(sd.surface, sd.roster, _adj_deny) is None
    # m↔n and a↔s are adjacent and not on the denylist. The gate that would
    # clear Willian / Patricis accepts the abstain twin.
    for kind, pair in (("collision_adj_mn", ("m", "n")), ("collision_adj_as", ("s", "a"))):
        row = grouped[kind][0]
        assert _diff(row.surface, row.roster[0]) == ("sub", *pair)
        assert _unique(row.surface, row.roster, _adj_deny) == row.roster[0]
        for twin in grouped[kind]:
            assert _unique(twin.surface, twin.roster, _adj_deny) == twin.roster[0]
    # a↔e and a↔o are the vowel typos and the absent names Williem / Mary Jone.
    for kind, pair in (("collision_vowel_ae", ("a", "e")), ("collision_vowel_oa", ("a", "o"))):
        row = grouped[kind][0]
        assert _diff(row.surface, row.roster[0]) == ("sub", *pair)
        assert not _adjacent(pair[0], pair[1])
        for twin in grouped[kind]:
            assert _unique(twin.surface, twin.roster, _edit1) == twin.roster[0]
    # A deletion lights prefix_ge3 and the edit gate still drops it.
    deletion = grouped["collision_del"][0]
    assert _diff(deletion.surface, deletion.roster[0])[0] == "del"
    cheap = _cheap(deletion.surface, deletion.roster[0])
    assert cheap["prefix_ge3"] == 1.0
    assert _ambiguous_edit(deletion.surface, deletion.roster[0])
    assert _featurize(deletion.surface, deletion.roster[0]) is None
    for twin in grouped["collision_del"]:
        assert _unique(twin.surface, twin.roster, _edit1) == twin.roster[0]
    # Equal-length substitutions have an empty cheap vector.
    for kind in (
        "collision_adj_sd", "collision_adj_mn", "collision_adj_as",
        "collision_vowel_ae", "collision_vowel_oa",
        "collision_homo", "collision_nose", "collision_diffpy", "collision_contain",
    ):
        row = grouped[kind][0]
        cheap = _cheap(row.surface, row.roster[0])
        assert all(cheap[key] == 0.0 for key in cheap), (kind, cheap)
        assert _ambiguous_edit(row.surface, row.roster[0])
    homo = grouped["collision_homo"][0]
    assert _pinyin_key(homo.surface) == _pinyin_key(homo.roster[0])
    assert _shared(homo.surface, homo.roster[0]) == "路"
    nose = grouped["collision_nose"][0]
    assert _pinyin_key(nose.surface) == _pinyin_key(nose.roster[0])
    assert _shared(nose.surface, nose.roster[0]) == ""
    diffpy = grouped["collision_diffpy"][0]
    assert _pinyin_key(diffpy.surface) != _pinyin_key(diffpy.roster[0])
    assert _shared(diffpy.surface, diffpy.roster[0]) == "帛"
    contain = grouped["collision_contain"][0]
    assert _pinyin_key(contain.surface) == _pinyin_key(contain.roster[0])
    assert _contains(contain.surface, contain.roster[0])
    assert _shared(contain.surface, contain.roster[0]) == ""
    for kind, pred in (
        ("collision_homo", lambda s, n: _pinyin_key(s) == _pinyin_key(n) and _shared(s, n) != ""),
        ("collision_nose", lambda s, n: _pinyin_key(s) == _pinyin_key(n) and _cjk_diff(s, n) and _shared(s, n) == ""),
        ("collision_diffpy", lambda s, n: _pinyin_key(s) != _pinyin_key(n) and _shared(s, n) != ""),
        ("collision_contain", _contains),
    ):
        for twin in grouped[kind]:
            assert _unique(twin.surface, twin.roster, pred) == twin.roster[0]


def test_ties_and_extensions_already_abstain():
    grouped = {}
    for row in build_rows():
        grouped.setdefault(row.kind, []).append(row)
    tie_en = grouped["tie_en"][0]
    dists = [name for name in tie_en.roster if _edit1(tie_en.surface, name)]
    assert dists == ["Rosmund", "Resmund"]
    assert _unique(tie_en.surface, tie_en.roster, _edit1) is None
    tie_zh = grouped["tie_zh"][0]
    assert [name for name in tie_zh.roster if _edit1(tie_zh.surface, name)] == ["郗璐", "郗露"]
    assert _unique(tie_zh.surface, tie_zh.roster, _edit1) is None
    for kind in ("hyphen", "space", "extension", "prefix", "middle", "superstring"):
        row = grouped[kind][0]
        assert row.label == "abstain"
        assert _extension(row.surface, row.roster[0])
        assert _unique(row.surface, row.roster, _edit1) is None
        assert best_name(row.surface, row.roster) is None


def test_typo_edit_dev_is_disjoint():
    rows = build_rows()
    own = questions(rows)
    residual = _residual_questions()
    typo = _typo_questions()
    lastname_q = _lastname_questions()
    bags = {
        "probe": probe_strings(),
        "surface": corpus_strings(),
        "abstain_dev": case_strings(build_cases(DEV_BANK)),
        "frozen": case_strings(build_cases(TEST_BANK)),
        "cjk": cjk_questions(),
        "nick": nick_questions(),
        "indirect": indirect_bank(),
        "indirect_q": indirect_questions(),
        "typo": typo,
        "cue": cue_questions(),
        "bare": bare_questions(),
        "residual": residual,
        "two_name": two_strings(),
        "focus": focus_strings(),
        "lastname": _LASTNAME_BANK,
    }
    assert hashlib.sha256("\n".join(typo).encode("utf-8")).hexdigest() == TYPO_SHA
    assert len(typo) == 68
    assert hashlib.sha256("\n".join(residual).encode("utf-8")).hexdigest() == RESIDUAL_SHA
    assert len(residual) == 100
    assert hashlib.sha256("\n".join(lastname_q).encode("utf-8")).hexdigest() == LASTNAME_Q_SHA
    assert len(lastname_q) == 101
    assert len(_LASTNAME_BANK) == 309
    for name, texts in bags.items():
        bag = set(texts)
        assert not any(text in bag for text in own), name
        score = 0.0
        for left in own:
            for right in texts:
                score = max(score, _jaccard(left, right))
        assert score < 0.50, (name, score)
    banned = set(CJK_NAMES)
    banned.update(_OLD_TYPO_NAMES)
    banned.update(_LASTNAME_NAMES)
    for lang in ("en", "zh"):
        banned.update(CUE_NAMES[lang])
        banned.update(BARE_NAMES[lang])
        banned.update(TWO_NAMES[lang])
        banned.update(FOCUS_NAMES[lang])
        for bank in (PEOPLE, ABSENT, FIT_NAMES, DEV_NAMES, ABSENT_NAMES):
            banned.update(bank[lang])
        banned.update(DEV_BANK["stored"][lang])
        banned.update(DEV_BANK["absent"][lang])
        banned.update(TEST_BANK["stored"][lang])
        banned.update(TEST_BANK["absent"][lang])
    banned.add(TWO_ABSENT["en"])
    banned.add(TWO_ABSENT["zh"])
    for row in nick_rows():
        banned.update(row.roster)
    for group in (SELF, OTHER, THIRD, INDIRECT_ABSENT):
        banned.update(group.values())
    for name in names_used(rows):
        assert name not in banned, name
    for text in own:
        for name in banned:
            assert not name_in_text(name, text), (name, text)


def test_matcher_does_not_import_the_typo_bank():
    align = Path("tidal/question_name_align.py").read_text(encoding="utf-8")
    surface = Path("tidal/question_surface.py").read_text(encoding="utf-8")
    data = Path("tidal/question_typo_edit_dev_data.py").read_text(encoding="utf-8")
    assert "question_typo_edit" not in align
    assert "question_typo_edit" not in surface
    assert "Tressis" not in align and "郗潞" not in surface
    assert "spikenard" not in align and "蠃" not in surface
    for blob in (align, surface):
        assert "enable_abstain" not in blob and "use_abstain" not in blob
        assert "llm_memory" not in blob and "LLMMemory" not in blob
    tree = ast.parse(data)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert set(imported) <= {"__future__", "hashlib", "dataclasses", "typing"}
