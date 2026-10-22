"""Surface questions for the held-out ``ask`` probe. Not training data.

``fit_question_head`` / ``build_facts`` never import this module. Training
questions are not sentences: each one is ``embed_event(_feats(name, rel, True),
name_unit + reversed(rel_unit))`` with integer name/rel ids. The strings here
are Chinese and English asks, paraphrases, cross-turn references, and
unanswerable questions. A validator checks the set is internally consistent
(unique asks, no answer text inside the ask or the stored utterance).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

Value = Union[str, Tuple[str, ...]]

RELS: Tuple[str, ...] = ("phone", "birthday", "places")

PEOPLE = {
    "en": ("Ada", "Mei", "Owen", "Priya", "Lena", "Noah", "Sofia", "Jules", "Hana", "Ravi", "Nico", "Iris"),
    "zh": ("林晓", "周宁", "陈可", "赵一楠", "苏晚", "顾清", "何安", "沈予", "江禾", "唐宁", "许迟", "梁夏"),
}
ABSENT = {
    "en": ("Clara", "Victor", "Edith", "Hassan", "Naomi", "Felix"),
    "zh": ("马岚", "宋北", "冯棠", "韩澈", "邓星", "曹野"),
}

# Words used only as ordered multi-token values. Kept out of the utterances
# below; the validator rejects a case whose gold token shows up in its ask
# or its stored sentence.
_PLACE_EN: Tuple[str, ...] = (
    "cinder", "vellum", "jasper", "quill", "thimble", "lodestone", "marrow", "pewter",
    "thistle", "cobble", "linden", "sorrel", "wicker", "flint", "yarrow", "cobnut",
    "tallow", "bramble", "spindle", "ochre", "larkspur", "hessian", "plinth", "coppice",
    "ferrule", "trunnion", "cleat", "scrip", "weft", "nock", "bole", "awn",
)
_PLACE_ZH: Tuple[str, ...] = (
    "青冈", "赤埠", "白汀", "墨池", "桐仓", "芦浦", "葛岭", "榆关",
    "柿岸", "槐市", "樟埠", "榕渡", "杉径", "枫闸", "柏亭", "兰渚",
    "菊畦", "竹坞", "梅坂", "荷塘", "梨沟", "枣林", "杏坡", "桃浦",
)

CHATS = {
    "en": (
        "haha wait that is not what I meant",
        "anyway the weather on the quay was awful",
        "sorry, go on, I was looking at the map",
    ),
    "zh": (
        "哈哈等等我说岔了",
        "码头今天风大得离谱",
        "你继续说，我刚才在看地图",
    ),
}

# family -> rel -> 12 templates. ``{name}`` is the only placeholder.
# primary: the utterance stored for a direct / indirect / distractor fact.
# cross: an earlier mention, stored before intervening facts.
# update_old / update_new: same identity, old value then the replacement.
_EN_PRIMARY = {
    "phone": (
        "{name} said the phone number to keep is the handset by the kiln.",
        "{name} uses a daytime mobile for the clinic rather than the flat.",
        "The line taped inside the saxophone case is {name}'s phone number.",
        "{name} asked that calls go to the receiver in the greenhouse.",
        "I wrote {name}'s telephone next to the night desk at the archive.",
        "{name} said the boat radio is the phone number for off shift.",
        "{name}'s phone number is the extension beside the lithography press.",
        "The contact card on the stairwell corkboard is {name}'s telephone.",
        "{name} told me the pottery-studio line is the phone to use.",
        "For {name}, store the phone number of the handset in the van.",
        "{name} left a phone number with the harbor office to pass along.",
        "The emergency handset in the red toolbox is {name}'s phone number.",
    ),
    "birthday": (
        "{name}'s birthday is the date their sister circles on the kitchen calendar.",
        "I was asked to remember {name}'s birthday, not the party the week after.",
        "The birthday on file for {name} is the day a card still arrives from their father.",
        "{name} told me which day counts as their birthday and to ignore the celebration.",
        "Put {name}'s date of birth down as the one they use on forms.",
        "{name}'s birthday, the one they actually claim, is the day we were told to keep.",
        "For {name} the birthday to store is the calendar day, not the weekend they host.",
        "{name} said their birthday is the morning their grandmother calls.",
        "The date of birth {name} gave me is the one in the school record.",
        "{name} asked that their birthday be the day on the passport, not the feast.",
        "I wrote {name}'s birthday beside the other family dates.",
        "{name}'s birthday is the day the bakery already has a note for.",
    ),
    "places": (
        "{name} listed the meeting spots and asked me to keep the order.",
        "The places {name} wants used for handoff were given as a short list.",
        "{name} named where to meet and said not to rearrange them.",
        "Store the stops {name} repeated, in the order they were spoken.",
        "{name}'s usual places are the ones they dictated, first to last.",
        "I kept the locations {name} gave me as an ordered list.",
        "{name} wants those spots remembered in sequence.",
        "The place list for {name} is short, and the order is part of it.",
        "{name} gave the locations once and told me not to sort them.",
        "For {name}, the places are a sequence they will ask for later.",
        "{name}'s places were spoken as a list, earliest stop first.",
        "Write down the places {name} uses, in the order they said them.",
    ),
}
_EN_CROSS = {
    "phone": (
        "Earlier {name} read a phone number off a napkin and then changed the subject.",
        "A few turns back {name} mentioned a mobile and moved on to the train.",
        "{name} gave a telephone before the other topics started.",
        "Before we drifted, {name} stated a phone number once.",
        "{name} dropped a contact number into the chat and then talked about lunch.",
        "The number {name} said out loud was a phone number, and then the subject changed.",
        "{name} recited a telephone early on and did not repeat it.",
        "Way earlier {name} left a phone number and the chat moved past it.",
        "{name} mentioned digits for a phone and we got onto something else.",
        "At the start {name} told us a telephone and then other facts piled up.",
        "{name} said a phone number in passing, before the rest of the details.",
        "First thing, {name} gave a telephone, and the conversation left it behind.",
    ),
    "birthday": (
        "Earlier {name} said a birthday and then the chat went elsewhere.",
        "{name} mentioned a date of birth a few turns ago and did not come back to it.",
        "Before the other facts, {name} told me when their birthday is.",
        "{name} dropped their birthday into the conversation and changed topic.",
        "A while back {name} stated a birth date, then we moved on.",
        "{name} gave a birthday early and the later turns were about other people.",
        "The day {name} was born came up once, earlier, and then we left it.",
        "{name} said their birthday before the distracting details started.",
        "Earlier on, {name} told us a calendar day for their birthday.",
        "{name} mentioned the birthday and immediately started another story.",
        "Back at the beginning {name} gave a date of birth and we talked over it.",
        "{name} stated a birthday in an earlier turn and has not repeated it.",
    ),
    "places": (
        "Earlier {name} listed places and then the subject changed.",
        "{name} named an ordered set of spots a few turns ago.",
        "Before the other facts, {name} gave the meeting places in order.",
        "{name} dictated locations earlier and we did not write them again.",
        "A while back {name} said where to meet, in a sequence, then moved on.",
        "{name} listed stops at the start and the chat filled up with other facts.",
        "The places {name} wanted came up once, earlier in the conversation.",
        "{name} gave the location list before we got distracted.",
        "Earlier {name} told me the spots in order and then talked about something else.",
        "{name} mentioned the places and the next turns were about other people.",
        "At the outset {name} listed where to go, first to last, and we left it.",
        "{name} said the meeting sequence in an earlier turn.",
    ),
}
_EN_OLD = {
    "phone": (
        "{name}'s phone number was the daytime landline at the front desk.",
        "The previous telephone for {name} was the lobby courtesy phone.",
        "{name} used to answer the workshop wall phone.",
        "Store {name}'s old phone number, the one for the loading dock.",
        "{name}'s former telephone is the shared handset by the copier.",
        "Until the change, {name}'s phone number was the gatehouse line.",
        "{name} said the old number was the phone in the supply closet.",
        "The retired telephone for {name} is the one at the service window.",
        "{name}'s previous phone number rings the basement office.",
        "Before the swap, {name}'s telephone was the pool phone.",
        "{name} told me their old phone number was the attic extension.",
        "The phone number {name} retired is the booth by the stairs.",
    ),
    "birthday": (
        "{name}'s birthday was recorded as the feast day, and that note is now wrong.",
        "The old birthday on file for {name} was the party date, not the real day.",
        "{name} used to cite a different birthday, the one on the faded form.",
        "We had {name}'s date of birth wrong: it was the form they discarded.",
        "{name}'s former birthday is the day the first card used.",
        "The previous birthday for {name} was the one their cousin remembered.",
        "{name} said to drop the old birthday, the weekday they no longer claim.",
        "An earlier note put {name}'s birthday on the rehearsal day.",
        "{name}'s old date of birth was the one printed on the spare badge.",
        "Before the correction, {name}'s birthday was the Sunday they host.",
        "{name} asked me to forget the previous birthday, the one in the draft.",
        "The retired birthday for {name} is the day the old calendar showed.",
    ),
    "places": (
        "{name}'s old places were a different ordered list, the one they retired.",
        "The previous stops for {name} are no longer the ones to use.",
        "{name} used to dictate another sequence of meeting spots.",
        "We had an older place list for {name}, and they withdrew it.",
        "{name}'s former locations were the draft sequence, not the current one.",
        "The earlier places for {name} should not be read back anymore.",
        "{name} replaced the old ordered spots with a new sequence.",
        "A previous list of places for {name} is out of date.",
        "{name}'s retired meeting sequence is not the one to repeat.",
        "Before the update, {name}'s places were the old handoff stops.",
        "{name} said the previous location list is cancelled.",
        "The outdated places for {name} were the ones in the first note.",
    ),
}
_EN_NEW = {
    "phone": (
        "{name}'s phone number is now the night handset in the annex.",
        "The telephone to use for {name} now is the mobile they just bought.",
        "{name} changed phones: the new number is the one in the blue case.",
        "Replace {name}'s telephone with the handset at the new bench.",
        "{name}'s current phone number is the direct line they opened today.",
        "From now on {name}'s phone number is the headset in the booth.",
        "{name} said the new phone is the one labeled with their initials.",
        "The updated telephone for {name} is the satellite handset.",
        "{name}'s phone number after the correction is the flat's new landline.",
        "Use {name}'s new telephone, the one they texted this morning.",
        "{name} switched the phone number to the spare they keep charged.",
        "The phone number on file for {name} is now the courtyard extension.",
    ),
    "birthday": (
        "{name}'s birthday is now the day written on the new passport.",
        "The corrected birthday for {name} is the date their mother confirmed.",
        "{name} fixed their birthday: keep the day on the replacement card.",
        "Update {name}'s date of birth to the one in the revised form.",
        "{name}'s birthday after the correction is the morning they named today.",
        "The current birthday for {name} is the date on the fresh certificate.",
        "{name} said to store the new birthday, not the one we had.",
        "A correction came in: {name}'s birthday is the day they just restated.",
        "{name}'s real birthday, as updated, is the date in the second letter.",
        "Replace {name}'s birthday with the day the registrar sent.",
        "{name} confirmed a new birthday and asked me to drop the old one.",
        "The birthday now on file for {name} is the date they spelled out again.",
    ),
    "places": (
        "{name}'s places are now a new ordered list they just dictated.",
        "The current stops for {name} replace the sequence we had.",
        "{name} gave an updated order of meeting spots.",
        "Store the new places for {name}, in the order they said this time.",
        "{name}'s locations after the change are the fresh sequence.",
        "The updated place list for {name} is the one from this morning.",
        "{name} restated the spots, and this order is the one to keep.",
        "A new sequence of places for {name} supersedes the old list.",
        "{name}'s meeting places are now the ones in the revised note.",
        "Replace {name}'s places with the order they repeated just now.",
        "{name} confirmed the new locations and cancelled the earlier list.",
        "The places on file for {name} are now the latest sequence.",
    ),
}
_ZH_PRIMARY = {
    "phone": (
        "{name}说要留的电话号码是画室门口那一部。",
        "{name}白天用诊所的手机，不用家里的座机。",
        "琴盒里贴着的那条线是{name}的电话号码。",
        "{name}让电话打到温室里的分机。",
        "我把{name}的电话号码记在档案室夜班台旁边。",
        "{name}说下班后的电话是船上的无线电。",
        "{name}的电话号码是石版室旁边的分机。",
        "楼梯间软木板上的联系卡是{name}的电话。",
        "{name}说陶艺工作室那条线才是该打的电话。",
        "给{name}存的电话号码是货车里的那部手机。",
        "{name}把电话号码留在港务室让他们转达。",
        "红色工具箱里的紧急手机是{name}的电话号码。",
    ),
    "birthday": (
        "{name}的生日是姐姐在厨房日历上圈出的那一天。",
        "要记住的是{name}的生日，不是一周后的聚会。",
        "{name}档案上的生日是父亲还会寄卡片的那一天。",
        "{name}说哪一天算生日，庆祝日不要算进去。",
        "把{name}的出生日期写成表格上用的那一天。",
        "{name}认的生日，就是让我们记下的那一天。",
        "{name}要存的生日是日历上的那天，不是请客的周末。",
        "{name}说生日是祖母打电话来的那个早晨。",
        "{name}给我的出生日期是学籍卡上的那一天。",
        "{name}要生日按护照上的日子记，不是节庆。",
        "我把{name}的生日写在其他家里人的日期旁边。",
        "{name}的生日是面包店已经注明的那一天。",
    ),
    "places": (
        "{name}报了见面的地方，让我按顺序记。",
        "{name}要用于交接的地点是一张短清单。",
        "{name}说了去哪里碰面，还说不要重排。",
        "按{name}重复的顺序把那些站点存下来。",
        "{name}常用的地方是亲口按先后说的那几个。",
        "我把{name}给的地点按顺序留着。",
        "{name}要这些地点按先后记住。",
        "{name}的地点清单不长，顺序也是内容的一部分。",
        "{name}只说了一次地点，让我不要排序打乱。",
        "{name}的这些地方是以后会问的一个序列。",
        "{name}的地点是按先到后到说出来的。",
        "把{name}要用的地方按原话的顺序写下来。",
    ),
}
_ZH_CROSS = {
    "phone": (
        "前头{name}从纸巾上念了一串电话号码，然后就换了话题。",
        "几轮之前{name}提过一部手机，接着去说火车的事。",
        "{name}在别的话题开始前给过一个电话。",
        "还没跑题的时候，{name}说过一次电话号码。",
        "{name}把联系电话丢进聊天里，然后去说午饭。",
        "{name}大声说的那串是电话号码，说完就换了题。",
        "{name}早先背过一个电话，后来没有再重复。",
        "更早的时候{name}留过电话号码，聊天已经翻篇了。",
        "{name}提过打电话用的数字，我们就去聊别的了。",
        "一开始{name}说了个电话，后面堆上了别的事实。",
        "{name}顺口说了个电话号码，在其余细节之前。",
        "{name}先给了电话，对话后来把它落在后面了。",
    ),
    "birthday": (
        "前头{name}说过生日，聊天随后去了别处。",
        "{name}几轮前提过出生日期，之后没有再回到这件事。",
        "在其他事实之前，{name}告诉我生日是哪一天。",
        "{name}把生日丢进对话里就换了话题。",
        "早些时候{name}说了一个出生日期，然后我们就往下聊了。",
        "{name}早早给了生日，后面几轮都在说别人。",
        "{name}哪一天出生只提前头说过一次，后来就放下了。",
        "{name}在那些岔开的细节之前说过自己的生日。",
        "更早的时候，{name}说过生日是日历上的哪一天。",
        "{name}提了生日，马上开始讲另一件事。",
        "开头{name}给过出生日期，后来的话把它盖过去了。",
        "{name}在更早的一轮说过生日，之后没有重复。",
    ),
    "places": (
        "前头{name}列过地方，然后话题就换了。",
        "{name}几轮前按顺序说过几个地点。",
        "在其他事实之前，{name}按顺序给了见面的地方。",
        "{name}更早口述过地点，后来没有再写一遍。",
        "早些时候{name}按顺序说了去哪里碰面，然后就翻篇了。",
        "{name}一开始列了站点，后面聊天堆满了别的事实。",
        "{name}要的那些地方只在前头出现过一次。",
        "{name}在我们走神之前给过地点清单。",
        "更早{name}按顺序告诉我地点，然后去说别的。",
        "{name}提过那些地方，后面几轮都在说别人。",
        "开头{name}按先后列了要去的地方，后来就放下了。",
        "{name}在更早的一轮说过见面的顺序。",
    ),
}
_ZH_OLD = {
    "phone": (
        "{name}的电话号码曾经是前台白天的座机。",
        "{name}以前的电话是大堂的公用电话。",
        "{name}过去接的是车间墙上的那部电话。",
        "把{name}的旧电话号码存成装卸台那一条。",
        "{name}先前的电话是复印机旁边的公用分机。",
        "在更换之前，{name}的电话号码是门房那条线。",
        "{name}说旧号码是储藏间里的电话。",
        "{name}停用的电话是服务窗口那一部。",
        "{name}以前的电话号码打到地下室办公室。",
        "换掉之前，{name}的电话是池边的那部。",
        "{name}告诉我旧电话号码是阁楼分机。",
        "{name}弃用的电话号码是楼梯口亭子里的那部。",
    ),
    "birthday": (
        "{name}的生日曾被记成节庆那天，那条记录现在是错的。",
        "{name}档案里的旧生日是聚会日，不是真正的那天。",
        "{name}以前说的是另一个生日，写在褪色表格上的那个。",
        "我们把{name}的出生日期记错了，那是已经作废的表格。",
        "{name}先前的生日是第一张卡片用过的那天。",
        "{name}以前的生日是表亲记得的那一天。",
        "{name}说把旧生日丢掉，那个他们不再认的工作日。",
        "更早的笔记把{name}的生日写成了排练那天。",
        "{name}的旧出生日期印在备用证件上。",
        "更正之前，{name}的生日是他们请客的那个星期日。",
        "{name}让我忘掉草稿里的那个旧生日。",
        "{name}弃用的生日是旧日历上印的那一天。",
    ),
    "places": (
        "{name}的旧地点是另一份已经弃用的顺序清单。",
        "{name}以前的站点不再是该用的那些。",
        "{name}过去口述的是另一套见面顺序。",
        "我们有过一份更老的{name}的地点清单，已经被收回。",
        "{name}先前的地点是草稿顺序，不是现在这份。",
        "{name}更早的地方不应该再被读出来。",
        "{name}用新的顺序换掉了旧的见面地点。",
        "{name}上一份地点清单已经过时。",
        "{name}弃用的见面顺序不是该复述的那份。",
        "更新之前，{name}的地方是旧的交接点。",
        "{name}说先前的地点清单作废了。",
        "{name}过时的地方是第一张纸条上的那些。",
    ),
}
_ZH_NEW = {
    "phone": (
        "{name}的电话号码现在是附楼里夜间的那部手机。",
        "{name}现在该打的电话是刚买的那部手机。",
        "{name}换了电话：新号码是蓝壳子里的那部。",
        "把{name}的电话换成新工作台旁边的分机。",
        "{name}现在的电话号码是今天开通的直线。",
        "从现在起{name}的电话号码是隔间里的耳机。",
        "{name}说新电话是标了姓名缩写的那一部。",
        "{name}更新后的电话是那部卫星手机。",
        "更正之后{name}的电话号码是公寓的新座机。",
        "用{name}的新电话，就是今天早上发过来的那部。",
        "{name}把电话号码换成了一直充着电的备用机。",
        "{name}现在档案上的电话号码是院子里的分机。",
    ),
    "birthday": (
        "{name}的生日现在是新护照上写的那一天。",
        "{name}更正后的生日是母亲确认的那个日期。",
        "{name}改了生日：记下替换卡片上的那一天。",
        "把{name}的出生日期更新成修订表格里的那天。",
        "更正之后{name}的生日是今天重新说的那个早晨。",
        "{name}现在的生日是新证明上的日期。",
        "{name}说要存新的生日，不是我们原来记的那个。",
        "来了一条更正：{name}的生日是刚刚重述的那一天。",
        "{name}更新后的真实生日是第二封信里的日期。",
        "把{name}的生日换成登记处寄来的那一天。",
        "{name}确认了新生日，并让我丢掉旧的。",
        "{name}现在档案上的生日是又拼了一遍的那个日期。",
    ),
    "places": (
        "{name}的地方现在是刚刚口述的一份新顺序。",
        "{name}现在的站点替换了我们原来的顺序。",
        "{name}给了一份更新过的见面顺序。",
        "把{name}的新地点按这一次说的顺序存下来。",
        "更换之后{name}的地点是这份新序列。",
        "{name}更新后的地点清单是今天早上的那份。",
        "{name}重说了地点，这一次的顺序才是要留的。",
        "{name}的一份新地点序列取代了旧清单。",
        "{name}的见面地点现在是修订笔记里的那些。",
        "把{name}的地方换成刚刚重复的那个顺序。",
        "{name}确认了新地点，并取消了更早的清单。",
        "{name}现在档案上的地方是最新的序列。",
    ),
}

_EN_Q_DIRECT = {
    "phone": (
        "What is {name}'s phone number?",
        "Which telephone belongs to {name}?",
        "Can you read me {name}'s phone number?",
        "I need {name}'s telephone number.",
        "What number is listed for {name}'s phone?",
        "Tell me {name}'s phone number.",
        "Which phone number is {name}'s?",
        "Do you have {name}'s phone?",
        "What's the phone on file for {name}?",
        "Give me {name}'s telephone.",
        "Where is {name}'s phone number written down?",
        "What digits make up {name}'s phone number?",
    ),
    "birthday": (
        "When is {name}'s birthday?",
        "What is {name}'s date of birth?",
        "What day was {name} born?",
        "{name}'s birth date?",
        "On what date was {name} born?",
        "Which day is {name}'s birthday?",
        "Tell me {name}'s birthday.",
        "What's the birthday on file for {name}?",
        "When was {name} born?",
        "What date should I put down as {name}'s birthday?",
        "Do you know {name}'s date of birth?",
        "Which calendar day is {name}'s birthday?",
    ),
    "places": (
        "Which places did {name} list?",
        "What is {name}'s list of places?",
        "Name the places {name} stored, in order.",
        "What places are on file for {name}?",
        "Read back the places {name} gave.",
        "Which ordered places belong to {name}?",
        "Tell me {name}'s places.",
        "What locations did {name} ask us to keep as places?",
        "What is the place list for {name}?",
        "Which places should I repeat for {name}?",
        "Give me the places {name} dictated.",
        "What places did {name} want stored?",
    ),
}
_EN_Q_INDIRECT = {
    "phone": (
        "How do I reach {name}?",
        "What's the best way to get hold of {name} tonight?",
        "If I needed to ring {name}, what would I dial?",
        "Which line connects me to {name}?",
        "How should I get in touch with {name} by voice?",
        "I lost the digits for {name}. What were they?",
        "What should I punch in to talk to {name}?",
        "Remind me how to dial {name}.",
        "Is there a handset I can use for {name}?",
        "Where do I call if I want {name}?",
        "What do I tap to call {name} from this phone?",
        "How can I place a voice call to {name}?",
    ),
    "birthday": (
        "When's the day people celebrate {name} getting older?",
        "If I were buying a cake for {name}, which date is the party?",
        "Which date is circled for {name} each year?",
        "When should I send {name} a card?",
        "What date is {name}'s yearly celebration?",
        "I forget the day {name} came into the world.",
        "When do we light candles for {name}?",
        "What should I write on {name}'s cake?",
        "The day {name} was brought home — when was that?",
        "Which day of the year belongs to {name}?",
        "When is the anniversary of the day {name} arrived?",
        "How do I know which morning is {name}'s?",
    ),
    "places": (
        "Where did {name} tell me to show up?",
        "Which spots did {name} write down?",
        "What locations are tied to {name}?",
        "Where should I go for {name}, in the order given?",
        "Which stops did {name} want used?",
        "Remind me where {name} said to meet.",
        "What sequence of spots did {name} leave?",
        "Where am I supposed to find {name}'s handoff points?",
        "Which sites did {name} name, first to last?",
        "What were the meeting points {name} chose?",
        "Where does {name} want the handoff to happen?",
        "Which waypoints did {name} leave in order?",
    ),
}
_EN_Q_CROSS = {
    "phone": (
        "That number {name} mentioned earlier, what was it?",
        "Can you repeat the digits {name} gave a few turns ago?",
        "What was the telephone {name} parked before the other topics?",
        "Remind me of the thing {name} said I should dial.",
        "What were those digits {name} read out before we changed subject?",
        "The line {name} brought up earlier — which one?",
        "Say again the number {name} mentioned and then moved past.",
        "A few turns back {name} gave a way to ring them. What was it?",
        "What was the handset {name} referred to before the digression?",
        "Pull back the digits {name} stated earlier in the chat.",
        "{name} mentioned a way to be rung and we talked over it. What was it?",
        "Earlier {name} gave a contact string. What was that number?",
    ),
    "birthday": (
        "That day {name} mentioned earlier, when was it?",
        "What birthday did {name} say a few turns ago?",
        "The date {name} dropped before we changed subject — what was it?",
        "Remind me of the calendar day {name} gave earlier.",
        "Which day did {name} name and then leave behind?",
        "What was the birth date {name} stated at the start?",
        "Say again the day {name} was born, the one from earlier.",
        "A few turns back {name} gave a birthday. What was it?",
        "What date did {name} mention before the other facts?",
        "Pull back the birthday {name} told us earlier.",
        "{name} mentioned a day and we moved on. When was it?",
        "Earlier {name} gave a date of birth. What was that day?",
    ),
    "places": (
        "Those places {name} mentioned earlier, what were they?",
        "What spots did {name} list a few turns ago?",
        "The locations {name} gave before the subject changed — what were they?",
        "Remind me of the sequence {name} dictated earlier.",
        "Which stops did {name} name and then leave behind?",
        "What was the place list {name} stated at the start?",
        "Say again where {name} said to meet, from earlier.",
        "A few turns back {name} gave locations. What were they?",
        "What sites did {name} mention before the other facts?",
        "Pull back the places {name} told us earlier.",
        "{name} mentioned stops and we moved on. What were they, in order?",
        "Earlier {name} gave a meeting sequence. What was it?",
    ),
}
_EN_Q_DISTRACTORS = {
    "phone": (
        "Setting the other details aside, what is {name}'s phone number?",
        "I still need {name}'s telephone after all of that.",
        "Ignore the side topics and read {name}'s phone number.",
        "With those other facts in the way, which telephone is {name}'s?",
        "Coming back to {name}, what phone number did we store?",
        "The extra names are a distraction: what is {name}'s phone?",
        "After those unrelated details, tell me {name}'s phone number.",
        "Don't use the other biographies — what is {name}'s phone number?",
        "Skipping the filler, what's {name}'s telephone?",
        "Those other records aren't the one I want. What is {name}'s phone?",
        "Past the decoys, which number is {name}'s phone?",
        "I asked about several people. What is {name}'s phone number?",
    ),
    "birthday": (
        "Setting the other details aside, when is {name}'s birthday?",
        "I still need {name}'s date of birth after all of that.",
        "Ignore the side topics and read {name}'s birthday.",
        "With those other facts in the way, which day was {name} born?",
        "Coming back to {name}, what birthday did we store?",
        "The extra names are a distraction: when is {name}'s birthday?",
        "After those unrelated details, tell me {name}'s birthday.",
        "Don't use the other biographies — what is {name}'s date of birth?",
        "Skipping the filler, when's {name}'s birthday?",
        "Those other records aren't the one I want. What day was {name} born?",
        "Past the decoys, which date is {name}'s birthday?",
        "I asked about several people. When is {name}'s birthday?",
    ),
    "places": (
        "Setting the other details aside, which places did {name} list?",
        "I still need {name}'s places after all of that.",
        "Ignore the side topics and read {name}'s places in order.",
        "With those other facts in the way, what places belong to {name}?",
        "Coming back to {name}, what place list did we store?",
        "The extra names are a distraction: what are {name}'s places?",
        "After those unrelated details, tell me {name}'s places.",
        "Don't use the other biographies — which places did {name} give?",
        "Skipping the filler, what is {name}'s place list?",
        "Those other records aren't the one I want. What places did {name} store?",
        "Past the decoys, which ordered places are {name}'s?",
        "I asked about several people. What places did {name} list?",
    ),
}
_EN_Q_UPDATE = {
    "phone": (
        "What number should I use for {name} now?",
        "Which telephone is current for {name}?",
        "{name} updated it — what is the phone number now?",
        "What's {name}'s phone after the change?",
        "The old line for {name} is gone. What do I dial now?",
        "What is {name}'s phone number these days?",
        "Which number replaced {name}'s previous phone?",
        "Give me {name}'s phone as updated.",
        "What is the new telephone for {name}?",
        "{name}'s phone changed. What is it?",
        "Which digits are current for {name}'s phone?",
        "What phone number is on file for {name} after the correction?",
    ),
    "birthday": (
        "What birthday should I use for {name} now?",
        "Which date of birth is current for {name}?",
        "{name} corrected it — when is the birthday now?",
        "What's {name}'s birthday after the change?",
        "The old day for {name} is gone. What date do I keep now?",
        "What is {name}'s birthday these days?",
        "Which date replaced {name}'s previous birthday?",
        "Give me {name}'s date of birth as updated.",
        "What is the new birthday for {name}?",
        "{name}'s birthday changed. What is it?",
        "Which day is current for {name}'s birthday?",
        "What birth date is on file for {name} after the correction?",
    ),
    "places": (
        "Which places should I use for {name} now?",
        "What place list is current for {name}?",
        "{name} updated it — what are the places now?",
        "What are {name}'s places after the change?",
        "The old stops for {name} are gone. What sequence do I keep now?",
        "What are {name}'s places these days?",
        "Which list replaced {name}'s previous places?",
        "Give me {name}'s places as updated.",
        "What is the new place list for {name}?",
        "{name}'s places changed. What are they?",
        "Which ordered spots are current for {name}?",
        "What places are on file for {name} after the correction?",
    ),
}
_EN_Q_MISSING = {
    "phone": (
        "Which phone number is filed under {name}?",
        "Did we write down a telephone for {name}?",
        "What phone, if any, belongs to {name}?",
        "Is there a stored phone number for {name}?",
        "Which telephone did {name} leave with us?",
        "What digits did we keep for {name}'s phone?",
    ),
    "birthday": (
        "Which birthday is filed under {name}?",
        "Did we write down a date of birth for {name}?",
        "What birthday, if any, belongs to {name}?",
        "Is there a stored birthday for {name}?",
        "Which birth date did {name} leave with us?",
        "What day did we keep for {name}'s birthday?",
    ),
    "places": (
        "Which places are filed under {name}?",
        "Did we write down places for {name}?",
        "What place list, if any, belongs to {name}?",
        "Is there a stored sequence of places for {name}?",
        "Which spots did {name} leave with us as places?",
        "What locations did we keep for {name}?",
    ),
}
_EN_Q_ABSENT = {
    "phone": (
        "Has anyone told us {name}'s phone number?",
        "Do we have a telephone for {name}?",
        "Which number would ring {name}?",
        "Is {name}'s phone in memory?",
        "What telephone belongs to {name}, if we ever heard one?",
        "Did {name} ever leave a phone number?",
    ),
    "birthday": (
        "Has anyone told us {name}'s birthday?",
        "Do we have a date of birth for {name}?",
        "Which day would be {name}'s birthday?",
        "Is {name}'s birthday in memory?",
        "What birth date belongs to {name}, if we ever heard one?",
        "Did {name} ever leave a birthday?",
    ),
    "places": (
        "Has anyone told us {name}'s places?",
        "Do we have a place list for {name}?",
        "Which spots would belong to {name}?",
        "Are {name}'s places in memory?",
        "What locations belong to {name}, if we ever heard them?",
        "Did {name} ever leave an ordered list of places?",
    ),
}

_ZH_Q_DIRECT = {
    "phone": (
        "{name}的电话号码是多少？",
        "{name}的手机号是哪一个？",
        "能把{name}的电话读给我吗？",
        "我要{name}的电话号码。",
        "{name}的电话留的是哪一串？",
        "告诉我{name}的手机号。",
        "{name}的电话号码是什么？",
        "你有{name}的电话吗？",
        "档案里{name}的电话是多少？",
        "把{name}的电话号码给我。",
        "{name}的手机号写在哪里？",
        "{name}的电话是由哪些数字组成的？",
    ),
    "birthday": (
        "{name}的生日是哪一天？",
        "{name}的出生日期是什么？",
        "{name}是哪一天出生的？",
        "{name}的生日？",
        "{name}出生在什么日期？",
        "{name}的生日是哪一天来着？",
        "告诉我{name}的生日。",
        "档案里{name}的生日是哪天？",
        "{name}是什么时候出生的？",
        "我该把{name}的生日写成哪一天？",
        "你知道{name}的出生日期吗？",
        "日历上{name}的生日是哪一天？",
    ),
    "places": (
        "{name}列出的地方有哪些？",
        "{name}的地点清单是什么？",
        "按顺序说出{name}存下的地方。",
        "档案里{name}的地方有哪些？",
        "把{name}给过的地点读出来。",
        "{name}的那些有序地点是哪些？",
        "告诉我{name}的地方。",
        "{name}让我们当作地点记下的是哪里？",
        "{name}的地点清单是什么内容？",
        "我该为{name}重复哪些地方？",
        "把{name}口述的地点给我。",
        "{name}想存下的地方有哪些？",
    ),
}
_ZH_Q_INDIRECT = {
    "phone": (
        "怎么联系{name}？",
        "今晚怎么找到{name}最靠谱？",
        "要是得给{name}打个电话，该拨哪一串？",
        "哪条线能接通{name}？",
        "想用语音找到{name}，该怎么做？",
        "我把{name}的数字弄丢了，是多少？",
        "要跟{name}说话，该按哪些键？",
        "提醒我怎么拨给{name}。",
        "有没有一部手机可以用来找{name}？",
        "想找{name}的话该往哪里打？",
        "从这部电话打给{name}要点哪里？",
        "怎么给{name}打一通语音电话？",
    ),
    "birthday": (
        "大家是在哪一天给{name}过成长的日子？",
        "要是给{name}买蛋糕，聚会该订在哪一天？",
        "每年圈给{name}的是哪一个日期？",
        "我该在哪一天给{name}寄卡片？",
        "{name}每年那次庆祝是哪一天？",
        "我忘了{name}来到世上的那一天。",
        "我们哪一天给{name}点蜡烛？",
        "{name}的蛋糕上该写哪一天？",
        "{name}被接回家的那天是什么时候？",
        "一年里哪一天是属于{name}的？",
        "{name}到来的那个纪念日是哪天？",
        "我怎么知道哪个早晨是{name}的？",
    ),
    "places": (
        "{name}让我去哪里露面？",
        "{name}写下的站点有哪些？",
        "跟{name}绑在一起的地点是哪里？",
        "按给过的顺序，我该为{name}去哪里？",
        "{name}想用的停靠点是哪些？",
        "提醒我{name}说在哪里碰面。",
        "{name}留下的地点序列是什么？",
        "我该去哪里找{name}的交接位置？",
        "{name}按先后点过的位置是哪些？",
        "{name}选的碰面点是哪里？",
        "{name}想把交接放在哪里？",
        "{name}按顺序留下的途经点是哪些？",
    ),
}
_ZH_Q_CROSS = {
    "phone": (
        "{name}前面提到的那个号码是多少？",
        "把{name}几轮前给的数字再重复一遍？",
        "{name}在其他话题之前放下的电话是哪一个？",
        "提醒我{name}说过该拨的那一串。",
        "{name}换话题前念出的数字是多少？",
        "{name}早先说的那条线是哪一条？",
        "再说一遍{name}提过又翻篇的那个号码。",
        "几轮之前{name}给过一个联系办法，是什么？",
        "{name}跑题前说的那部手机是哪一部？",
        "把{name}在聊天里早先说的数字找回来。",
        "{name}提过怎么打给自己，后来被别的话盖住了，是什么？",
        "{name}更早给过一串联系方式，那个号码是多少？",
    ),
    "birthday": (
        "{name}前面提到的那一天是什么时候？",
        "{name}几轮前说的生日是哪一天？",
        "{name}换话题前丢下的日期是哪一天？",
        "提醒我{name}早先给的那个日历日。",
        "{name}说过又放下的是哪一天？",
        "{name}一开始说的出生日期是什么？",
        "再说一遍{name}更早说过的出生那天。",
        "几轮之前{name}给过一个生日，是哪一天？",
        "{name}在其他事实之前提到的日期是什么？",
        "把{name}早先告诉我的生日找回来。",
        "{name}提过某一天我们就翻篇了，是哪一天？",
        "{name}更早给过出生日期，那天是什么时候？",
    ),
    "places": (
        "{name}前面提到的那些地方是哪里？",
        "{name}几轮前列过的地点有哪些？",
        "{name}换话题前给的位置是哪些？",
        "提醒我{name}早先口述的顺序。",
        "{name}说过又放下的站点是哪些？",
        "{name}一开始说的地点清单是什么？",
        "再说一遍{name}更早说的碰面地点。",
        "几轮之前{name}给过地点，是哪些？",
        "{name}在其他事实之前提到的位置是哪里？",
        "把{name}早先告诉我的地方找回来。",
        "{name}提过一些站点我们就翻篇了，按顺序是哪些？",
        "{name}更早给过见面顺序，是什么？",
    ),
}
_ZH_Q_DISTRACTORS = {
    "phone": (
        "先把其他细节放下，{name}的电话号码是多少？",
        "说了这么多，我还是要{name}的电话。",
        "别管旁边的话题，把{name}的电话号码读出来。",
        "那些事实挡在中间，{name}的电话是哪一部？",
        "回到{name}，我们存下的电话号码是多少？",
        "别的名字都是干扰：{name}的电话是什么？",
        "在那些无关的细节之后，告诉我{name}的电话号码。",
        "不要用别人的履历——{name}的电话号码是多少？",
        "跳过那些填充，{name}的电话是什么？",
        "其他记录不是我要的。{name}的电话是多少？",
        "过了那些诱饵，{name}的电话号码是哪一串？",
        "我问了好几个人。{name}的电话号码是多少？",
    ),
    "birthday": (
        "先把其他细节放下，{name}的生日是哪一天？",
        "说了这么多，我还是要{name}的出生日期。",
        "别管旁边的话题，把{name}的生日读出来。",
        "那些事实挡在中间，{name}是哪一天出生的？",
        "回到{name}，我们存下的生日是哪一天？",
        "别的名字都是干扰：{name}的生日是什么时候？",
        "在那些无关的细节之后，告诉我{name}的生日。",
        "不要用别人的履历——{name}的出生日期是什么？",
        "跳过那些填充，{name}的生日是哪天？",
        "其他记录不是我要的。{name}是哪一天出生的？",
        "过了那些诱饵，{name}的生日是哪一个日期？",
        "我问了好几个人。{name}的生日是哪一天？",
    ),
    "places": (
        "先把其他细节放下，{name}列出的地方有哪些？",
        "说了这么多，我还是要{name}的地点。",
        "别管旁边的话题，按顺序把{name}的地方读出来。",
        "那些事实挡在中间，{name}的地方是哪些？",
        "回到{name}，我们存下的地点清单是什么？",
        "别的名字都是干扰：{name}的地方有哪些？",
        "在那些无关的细节之后，告诉我{name}的地点。",
        "不要用别人的履历——{name}给过哪些地方？",
        "跳过那些填充，{name}的地点清单是什么？",
        "其他记录不是我要的。{name}存了哪些地方？",
        "过了那些诱饵，{name}的有序地点是哪些？",
        "我问了好几个人。{name}列出的地方有哪些？",
    ),
}
_ZH_Q_UPDATE = {
    "phone": (
        "现在该用{name}的哪一个号码？",
        "{name}当前的电话是哪一部？",
        "{name}更新过了——电话号码现在是多少？",
        "更换之后{name}的电话是什么？",
        "{name}的旧线路没了。现在该拨哪一串？",
        "{name}这阵子的电话号码是多少？",
        "哪一串换成了{name}以前的电话？",
        "把{name}更新后的电话给我。",
        "{name}的新电话是什么？",
        "{name}的电话改了。现在是多少？",
        "{name}的电话现在是哪些数字？",
        "更正之后档案里{name}的电话号码是多少？",
    ),
    "birthday": (
        "现在该用{name}的哪个生日？",
        "{name}当前的出生日期是哪一天？",
        "{name}更正过了——生日现在是哪一天？",
        "更换之后{name}的生日是什么？",
        "{name}的旧日子没了。现在该留哪一天？",
        "{name}这阵子的生日是哪一天？",
        "哪一天换成了{name}以前的生日？",
        "把{name}更新后的出生日期给我。",
        "{name}的新生日是什么？",
        "{name}的生日改了。现在是哪一天？",
        "{name}的生日现在是哪一天？",
        "更正之后档案里{name}的生日是哪一天？",
    ),
    "places": (
        "现在该用{name}的哪些地方？",
        "{name}当前的地点清单是什么？",
        "{name}更新过了——地方现在有哪些？",
        "更换之后{name}的地点是哪些？",
        "{name}的旧站点没了。现在该留哪一套顺序？",
        "{name}这阵子的地方有哪些？",
        "哪一份清单换成了{name}以前的地点？",
        "把{name}更新后的地方给我。",
        "{name}的新地点清单是什么？",
        "{name}的地方改了。现在是哪些？",
        "{name}现在的有序地点是哪些？",
        "更正之后档案里{name}的地方有哪些？",
    ),
}
_ZH_Q_MISSING = {
    "phone": (
        "档案里归在{name}名下的电话号码是哪一串？",
        "我们有没有写下{name}的电话？",
        "{name}如果有电话，是哪一部？",
        "存过{name}的电话号码吗？",
        "{name}留下的电话是哪一个？",
        "我们为{name}的电话留了哪些数字？",
    ),
    "birthday": (
        "档案里归在{name}名下的生日是哪一天？",
        "我们有没有写下{name}的出生日期？",
        "{name}如果有生日，是哪一天？",
        "存过{name}的生日吗？",
        "{name}留下的出生日期是哪一天？",
        "我们为{name}的生日留了哪一天？",
    ),
    "places": (
        "档案里归在{name}名下的地方有哪些？",
        "我们有没有写下{name}的地点？",
        "{name}如果有地点清单，是哪些？",
        "存过{name}的地点序列吗？",
        "{name}作为地点留下的站点是哪些？",
        "我们为{name}留了哪些位置？",
    ),
}
_ZH_Q_ABSENT = {
    "phone": (
        "有人告诉过我们{name}的电话号码吗？",
        "我们有{name}的电话吗？",
        "哪一串号码能打到{name}？",
        "{name}的电话在记忆里吗？",
        "如果我们听过，{name}的电话是哪一部？",
        "{name}留下过电话号码吗？",
    ),
    "birthday": (
        "有人告诉过我们{name}的生日吗？",
        "我们有{name}的出生日期吗？",
        "哪一天会是{name}的生日？",
        "{name}的生日在记忆里吗？",
        "如果我们听过，{name}的出生日期是哪一天？",
        "{name}留下过生日吗？",
    ),
    "places": (
        "有人告诉过我们{name}的地方吗？",
        "我们有{name}的地点清单吗？",
        "哪些地点会属于{name}？",
        "{name}的地方在记忆里吗？",
        "如果我们听过，{name}的位置是哪些？",
        "{name}留下过按顺序的地点吗？",
    ),
}

STATEMENTS = {
    "en": {"primary": _EN_PRIMARY, "cross": _EN_CROSS, "update_old": _EN_OLD, "update_new": _EN_NEW},
    "zh": {"primary": _ZH_PRIMARY, "cross": _ZH_CROSS, "update_old": _ZH_OLD, "update_new": _ZH_NEW},
}
QUESTIONS = {
    "en": {
        "direct": _EN_Q_DIRECT,
        "indirect": _EN_Q_INDIRECT,
        "cross": _EN_Q_CROSS,
        "distractors": _EN_Q_DISTRACTORS,
        "update": _EN_Q_UPDATE,
        "missing": _EN_Q_MISSING,
        "absent": _EN_Q_ABSENT,
    },
    "zh": {
        "direct": _ZH_Q_DIRECT,
        "indirect": _ZH_Q_INDIRECT,
        "cross": _ZH_Q_CROSS,
        "distractors": _ZH_Q_DISTRACTORS,
        "update": _ZH_Q_UPDATE,
        "missing": _ZH_Q_MISSING,
        "absent": _ZH_Q_ABSENT,
    },
}

CATEGORIES: Tuple[str, ...] = (
    "en_direct",
    "zh_direct",
    "en_indirect",
    "zh_indirect",
    "en_cross_turn",
    "zh_cross_turn",
    "en_after_distractors",
    "zh_after_distractors",
    "en_after_update",
    "zh_after_update",
    "en_unanswerable",
    "zh_unanswerable",
)

REQUIRED_QUESTIONS: Tuple[str, ...] = (
    "How do I reach you?",
    "When's your birthday?",
    "What day were you born?",
    "that number you mentioned earlier",
    "我怎么联系你？",
    "你的生日是哪一天？",
    "你是哪一天出生的？",
    "你刚才说的那个号码是多少？",
)


@dataclass(frozen=True)
class Write:
    text: str
    value: Value
    name: str
    rel: str
    kind: str = "fact"


@dataclass(frozen=True)
class Case:
    category: str
    lang: str
    rel: str
    name: str
    question: str
    frame: str
    gold: Optional[Value]
    old: Optional[Value]
    statement: str
    writes: Tuple[Write, ...]
    subtype: str = ""


def _serial(lang: str, pi: int, rel: str, version: int) -> int:
    li = 0 if lang == "en" else 1
    return (((li * len(PEOPLE["en"]) + pi) * len(RELS) + RELS.index(rel)) * 4 + version)


def _value(lang: str, pi: int, rel: str, version: int) -> Value:
    serial = _serial(lang, pi, rel, version)
    if rel == "phone":
        return f"+44-{300 + (serial % 500):03d}-{1000 + serial:04d}-{4000 + (serial * 3) % 10000:04d}"
    if rel == "birthday":
        year = 1920 + serial
        month = (serial % 12) + 1
        day = (serial % 27) + 1
        kind = serial % 5
        if kind == 0:
            return f"{year:04d}-{month:02d}-{day:02d}"
        if kind == 1:
            return f"{month:02d}/{day:02d}/{year:04d}"
        if kind == 2:
            return f"{day:02d}.{month:02d}.{year:04d}"
        if kind == 3:
            months = (
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December",
            )
            return f"{day} {months[month - 1]} {year}"
        return f"{year}年{month}月{day}日"
    words = _PLACE_EN if lang == "en" else _PLACE_ZH
    n = len(words)
    # Mixed radix into an ordered triple of distinct words. Injective for
    # serial < n*(n-1)*(n-2), which covers every (lang, person, rel, version).
    a = serial % n
    rest = serial // n
    b = [i for i in range(n) if i != a][rest % (n - 1)]
    rest //= n - 1
    c = [i for i in range(n) if i != a and i != b][rest % (n - 2)]
    return (words[a], words[b], words[c])


def _fact(lang: str, pi: int, rel: str, version: int, family: str) -> Write:
    name = PEOPLE[lang][pi]
    text = STATEMENTS[lang][family][rel][pi].format(name=name)
    return Write(text, _value(lang, pi, rel, version), name, rel, "fact")


def _others(lang: str, pi: int, rel: str, n: int) -> Tuple[Tuple[int, str], ...]:
    ring = [(p, r) for p in range(len(PEOPLE[lang])) for r in RELS]
    start = ring.index((pi, rel))
    out = []
    step = 1
    while len(out) < n:
        item = ring[(start + step) % len(ring)]
        step += 1
        if item != (pi, rel):
            out.append(item)
        if step > len(ring) + 2:
            raise RuntimeError("distractor ring exhausted")
    return tuple(out)


def _chats(lang: str) -> Tuple[Write, ...]:
    return tuple(Write(text, "chat-not-stored", "chatter", "aside", "chat") for text in CHATS[lang])


def _case(
    category: str,
    lang: str,
    rel: str,
    name: str,
    question: str,
    frame: str,
    gold: Optional[Value],
    old: Optional[Value],
    statement: str,
    writes: Sequence[Write],
    subtype: str = "",
) -> Case:
    return Case(category, lang, rel, name, question, frame, gold, old, statement, tuple(writes), subtype)


def _named(lang: str, pi: int, rel: str, style: str, category: str, family: str, version: int,
           n_dist: int, *, gold_first: bool = False, chats: bool = False, trailing: int = 0,
           subtype: str = "", old: Optional[Write] = None) -> Case:
    name = PEOPLE[lang][pi]
    frame = QUESTIONS[lang][style][rel][pi]
    question = frame.format(name=name)
    gold_w = _fact(lang, pi, rel, version, family)
    dists = [_fact(lang, p, r, 0, "primary") for p, r in _others(lang, pi, rel, n_dist)]
    writes: list = []
    if old is not None:
        writes.extend(dists)
        writes.append(old)
        writes.append(gold_w)
    elif gold_first:
        writes.append(gold_w)
        writes.extend(dists)
    elif chats:
        mid = n_dist - trailing
        writes.extend(dists[:mid])
        writes.extend(_chats(lang))
        writes.append(gold_w)
        writes.extend(dists[mid:])
    else:
        writes.extend(dists)
        writes.append(gold_w)
    return _case(
        category, lang, rel, name, question, frame, gold_w.value,
        None if old is None else old.value, gold_w.text, writes, subtype,
    )


def build_cases() -> Tuple[Case, ...]:
    """Every held-out case. Deterministic. No draw from the training RNG."""
    cases = []
    for lang in ("en", "zh"):
        for pi in range(len(PEOPLE[lang])):
            for rel in RELS:
                cases.append(_named(lang, pi, rel, "direct", f"{lang}_direct", "primary", 0, 4))
                cases.append(_named(lang, pi, rel, "indirect", f"{lang}_indirect", "primary", 0, 4))
                cases.append(_named(
                    lang, pi, rel, "cross", f"{lang}_cross_turn", "cross", 0, 5, gold_first=True,
                ))
                cases.append(_named(
                    lang, pi, rel, "distractors", f"{lang}_after_distractors", "primary", 0, 10,
                    chats=True, trailing=2,
                ))
                old = _fact(lang, pi, rel, 1, "update_old")
                cases.append(_named(
                    lang, pi, rel, "update", f"{lang}_after_update", "update_new", 2, 4, old=old,
                ))
        for pi in range(len(ABSENT[lang])):
            for rel in RELS:
                cases.append(_missing(lang, pi, rel))
                cases.append(_absent(lang, pi, rel))
        cases.extend(_extras(lang))
    return tuple(cases)


def _missing(lang: str, pi: int, rel: str) -> Case:
    name = PEOPLE[lang][pi]
    frame = QUESTIONS[lang]["missing"][rel][pi]
    writes = [_fact(lang, pi, r, 0, "primary") for r in RELS if r != rel]
    for p, r in _others(lang, pi, rel, 8):
        if p == pi:
            continue
        writes.append(_fact(lang, p, r, 0, "primary"))
        if len(writes) >= 6:
            break
    return _case(
        f"{lang}_unanswerable", lang, rel, name, frame.format(name=name), frame,
        None, None, "", writes, "missing_rel",
    )


def _absent(lang: str, pi: int, rel: str) -> Case:
    name = ABSENT[lang][pi]
    frame = QUESTIONS[lang]["absent"][rel][pi]
    ring = [(p, r) for p in range(len(PEOPLE[lang])) for r in RELS]
    writes = [_fact(lang, p, r, 0, "primary") for p, r in ring[pi:pi + 6]]
    return _case(
        f"{lang}_unanswerable", lang, rel, name, frame.format(name=name), frame,
        None, None, "", writes, "absent_person",
    )


def _extras(lang: str) -> Tuple[Case, ...]:
    """Second-person asks and one unnamed cross-turn reference per language.

    The stored utterance is first person. The identity is still the persona
    the rest of that language's facts use as ``people[0]``.
    """
    pi = 0
    name = PEOPLE[lang][pi]
    if lang == "en":
        specs = (
            ("en_indirect", "phone", "How do I reach you?",
             "My phone number is the handset I actually answer.", "second_person", False),
            ("en_indirect", "birthday", "When's your birthday?",
             "The day I was born is the one I asked you to keep.", "second_person", False),
            ("en_indirect", "birthday", "What day were you born?",
             "The day I was born is the one I asked you to keep.", "second_person", False),
            ("en_indirect", "places", "Which spots should I use when I come to you?",
             "I listed the meeting spots in the order I want them kept.", "second_person", False),
            ("en_cross_turn", "phone", "that number you mentioned earlier",
             "Earlier I read my phone number off a napkin and then changed the subject.",
             "anaphora_unnamed", True),
        )
    else:
        specs = (
            ("zh_indirect", "phone", "我怎么联系你？",
             "我的电话号码是我真正会接的那一部。", "second_person", False),
            ("zh_indirect", "birthday", "你的生日是哪一天？",
             "我出生的日子就是我让你记住的那一天。", "second_person", False),
            ("zh_indirect", "birthday", "你是哪一天出生的？",
             "我出生的日子就是我让你记住的那一天。", "second_person", False),
            ("zh_indirect", "places", "我去找你的话该在哪些地点碰面？",
             "我按想保留的顺序报过见面的地点。", "second_person", False),
            ("zh_cross_turn", "phone", "你刚才说的那个号码是多少？",
             "我刚才从纸巾上念了一串电话号码，然后就换了话题。",
             "anaphora_unnamed", True),
        )
    out = []
    for category, rel, question, statement, subtype, gold_first in specs:
        gold = _value(lang, pi, rel, 0)
        dists = [_fact(lang, p, r, 0, "primary") for p, r in _others(lang, pi, rel, 5 if gold_first else 4)]
        gold_w = Write(statement, gold, name, rel, "fact")
        writes = (gold_w, *dists) if gold_first else (*dists, gold_w)
        out.append(_case(category, lang, rel, name, question, question, gold, None, statement, writes, subtype))
    return tuple(out)


def _tokens(text: str) -> set:
    import re
    out = set()
    for match in re.finditer(r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:'[A-Za-z]+)?", text):
        seg = match.group(0)
        if "\u4e00" <= seg[0] <= "\u9fff":
            out.add(seg)
        else:
            out.add(seg.lower())
    return out


def _leaks(text: str, value: Optional[Value]) -> bool:
    if value is None or text == "":
        return False
    if isinstance(value, str):
        return value.lower() in text.lower()
    toks = _tokens(text)
    return any(piece.lower() in toks for piece in value)


def validate_corpus(cases: Sequence[Case]) -> Tuple[str, ...]:
    """Structural problems in the probe. Empty means the set is usable."""
    errors = []
    if len(cases) < 300:
        errors.append(f"only {len(cases)} cases")
    by_cat: dict = {}
    questions = []
    frames: dict = {}
    values_global = []
    for case in cases:
        by_cat.setdefault(case.category, 0)
        by_cat[case.category] += 1
        frames.setdefault(case.category, set()).add(case.frame)
        questions.append(case.question)
        if case.question.strip() != case.question or case.question == "":
            errors.append(f"blank question in {case.category}")
        if _leaks(case.question, case.gold) or _leaks(case.statement, case.gold):
            errors.append(f"gold leaked into text: {case.category} {case.question}")
        if _leaks(case.question, case.old) or _leaks(case.statement, case.old):
            errors.append(f"old value leaked into text: {case.category} {case.question}")
        fact_writes = [w for w in case.writes if w.kind == "fact"]
        if len({w.text for w in fact_writes}) != len(fact_writes):
            errors.append(f"duplicate statement in {case.category} {case.question}")
        seen_values = []
        for w in fact_writes:
            if w.value in seen_values:
                errors.append(f"duplicate value in {case.category} {case.question}")
                break
            seen_values.append(w.value)
            values_global.append(w.value)
        ids = [(w.name, w.rel) for w in fact_writes]
        if case.gold is None:
            if (case.name, case.rel) in ids:
                errors.append(f"unanswerable stored the asked fact: {case.question}")
            if case.subtype not in ("missing_rel", "absent_person"):
                errors.append(f"unanswerable subtype {case.subtype}")
        else:
            matched = [w for w in fact_writes if w.name == case.name and w.rel == case.rel and w.value == case.gold]
            if len(matched) != 1:
                errors.append(f"gold write missing: {case.category} {case.question}")
            if case.statement == "":
                errors.append(f"answerable case lacks a statement: {case.question}")
        if case.category.endswith("after_update"):
            if case.old is None or case.old == case.gold:
                errors.append(f"update without a distinct old value: {case.question}")
            if not any(w.value == case.old for w in fact_writes):
                errors.append(f"old value not written: {case.question}")
        elif case.old is not None:
            errors.append(f"old value outside updates: {case.question}")
        if case.category.endswith("after_distractors"):
            if sum(1 for w in case.writes if w.kind == "chat") != 3:
                errors.append(f"distractor case chats: {case.question}")
        elif any(w.kind == "chat" for w in case.writes):
            errors.append(f"chat outside distractors: {case.question}")
        if case.question == case.statement:
            errors.append(f"question copies the statement: {case.question}")
    for cat in CATEGORIES:
        n = by_cat.get(cat, 0)
        if n < 24:
            errors.append(f"{cat} has {n} cases")
        if len(frames.get(cat, ())) < 12:
            errors.append(f"{cat} has only {len(frames.get(cat, ()))} frames")
    if len(questions) != len(set(questions)):
        dup = sorted({q for q in questions if questions.count(q) > 1})
        errors.append(f"duplicate questions: {dup[:4]}")
    for q in REQUIRED_QUESTIONS:
        if q not in questions:
            errors.append(f"missing required question: {q}")
    for lang in ("en", "zh"):
        for person in ABSENT[lang]:
            if person in PEOPLE[lang]:
                errors.append(f"absent name collides with a stored persona: {person}")
        rendered = []
        for family, rels in STATEMENTS[lang].items():
            for rel, bank in rels.items():
                if len(bank) != 12:
                    errors.append(f"{lang} {family} {rel} has {len(bank)} statements")
                for pi, tmpl in enumerate(bank):
                    rendered.append(tmpl.format(name=PEOPLE[lang][pi]))
        if len(rendered) != len(set(rendered)):
            errors.append(f"duplicate statement template render in {lang}")
        for style, rels in QUESTIONS[lang].items():
            expect = 6 if style in ("missing", "absent") else 12
            for rel, bank in rels.items():
                if len(bank) != expect:
                    errors.append(f"{lang} {style} {rel} has {len(bank)} questions, want {expect}")
    # Global value identity: the same (lang, person, rel, version) is reused on
    # purpose. Distinct serials must not print the same phone, date, or triple.
    serial_values = []
    for lang in ("en", "zh"):
        for pi in range(len(PEOPLE[lang])):
            for rel in RELS:
                for version in (0, 1, 2):
                    serial_values.append(_value(lang, pi, rel, version))
    if len(serial_values) != len(set(serial_values)):
        errors.append("two different serials produced the same stored value")
    return tuple(errors)


def corpus_summary(cases: Optional[Sequence[Case]] = None) -> dict:
    cases = build_cases() if cases is None else cases
    errors = validate_corpus(cases)
    by_cat = {}
    frames = {}
    subs = {}
    for case in cases:
        by_cat[case.category] = by_cat.get(case.category, 0) + 1
        frames.setdefault(case.category, set()).add(case.frame)
        if case.subtype:
            key = f"{case.category}/{case.subtype}"
            subs[key] = subs.get(key, 0) + 1
    return {
        "n": len(cases),
        "errors": errors,
        "by_category": by_cat,
        "frames": {k: len(v) for k, v in frames.items()},
        "subtypes": subs,
    }


def probe_strings(cases: Optional[Sequence[Case]] = None) -> Tuple[str, ...]:
    """Unique surface strings the encoder will see, in first-seen order."""
    cases = build_cases() if cases is None else cases
    seen = []
    bag = set()
    for case in cases:
        texts = [case.question, case.statement]
        texts.extend(w.text for w in case.writes)
        for text in texts:
            if text and text not in bag:
                bag.add(text)
                seen.append(text)
    return tuple(seen)
