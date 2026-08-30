"""A 100-question evaluation suite written in Saudi colloquial Arabic.

The knowledge base is written in Modern Standard Arabic. Real users are not:
they type Najdi/Hijazi dialect. This suite exists to measure that specific gap,
because it is invisible to the MSA-phrased core suite in :mod:`faqrag.eval_data`.

The dialect features that actually break retrieval are:

* **Interrogatives** -- ``وش`` / ``ايش`` / ``وشو`` for ``ما``, ``وين`` for ``أين``,
  ``ليش`` for ``لماذا``, ``مين`` for ``من``, ``بكم`` for ``كم سعر``. None of these
  tokens appear anywhere in the corpus, so BM25 gets no signal from the
  question word at all and the dense retriever carries the query alone.
* **Verbs** -- ``أبغى`` / ``أبي`` / ``ودي`` / ``بغيت`` for ``أريد``,
  ``أقدر`` / ``يمديني`` for ``أستطيع``, ``تسوون`` for ``تقدمون``,
  ``تبون`` for ``تريدون``.
* **Particles and fillers** -- ``فيه`` / ``ما فيه`` for ``هل يوجد``, ``عشان`` for
  ``لكي``, ``اللي`` for ``الذي``, ``حقكم`` / ``حقي`` as possessives, ``الحين``
  for ``الآن``, and a general absence of ``هل`` before yes/no questions.
* **Orthography** -- users drop hamza (``اقدر`` for ``أقدر``) and write ``ه``
  for ``ة``. :func:`faqrag.lang.normalise` folds these, which is why that
  normalisation is load-bearing rather than cosmetic.

Coverage: all 26 FAQs, plus questions that are deliberately unanswerable so the
refusal path is measured on dialect input too -- pricing, cancellation, refunds,
opening hours, and coverage outside Saudi Arabia appear nowhere in the corpus.
"""

from __future__ import annotations

from .eval_data import CONTACT_PATTERN, EvalCase

SAUDI_CASES: tuple[EvalCase, ...] = (
    # ================= عن موفق  (FAQ 001-006) =================
    EvalCase("وش هي موفق؟", "ar", "clean", ("001",)),
    EvalCase("ايش تسوي منصة موفق بالضبط؟", "ar", "clean", ("001",)),
    EvalCase(
        "موفق هذي وش قصتها وش تقدم؟", "ar", "ambiguous", ("001", "004"),
        note="Reads as either 'what is it' or 'what is its story'.",
    ),
    EvalCase("وش هدفكم من المنصة؟", "ar", "ambiguous", ("001", "002", "004")),
    EvalCase("وش رؤيتكم للمستقبل؟", "ar", "clean", ("002",)),
    EvalCase("وش القيم اللي تشتغلون عليها؟", "ar", "clean", ("003",)),
    EvalCase("ليش أسستوا موفق من الأساس؟", "ar", "clean", ("004",)),
    EvalCase("وش قصة تأسيس الشركة؟", "ar", "clean", ("004",)),
    EvalCase("كم مركز طبي معتمد عندكم؟", "ar", "clean", ("005",)),
    EvalCase("كم مدينة تغطون بالمملكة؟", "ar", "clean", ("005",)),
    EvalCase("وش إنجازاتكم لين الحين؟", "ar", "clean", ("005",)),
    EvalCase("مين المدير التنفيذي حقكم؟", "ar", "clean", ("006",)),
    EvalCase("وش قال الرئيس التنفيذي عن موفق؟", "ar", "clean", ("006",)),

    # ================= خدمات الأفراد  (FAQ 007-009) =================
    EvalCase("وش الفحوصات اللي تسوونها للأفراد؟", "ar", "clean", ("007",)),
    EvalCase("عندكم فحص إقامة؟", "ar", "clean", ("007",)),
    EvalCase(
        "أبغى أجدد إقامتي، وش الفحص اللي أحتاجه؟", "ar", "paraphrase", ("007", "009"),
        note="Task phrased as a personal situation rather than a service name.",
    ),
    EvalCase("فيه فحص رخصة قيادة عندكم؟", "ar", "clean", ("007",)),
    EvalCase("أبغى أسوي فحص قبل الزواج", "ar", "paraphrase", ("007", "009")),
    EvalCase("عندكم فحص الشهادة الصحية؟", "ar", "clean", ("007",)),
    EvalCase("فيه فحص لمناديب التوصيل؟", "ar", "clean", ("007",)),
    EvalCase("أبغى فحص قبل ما يدخل ولدي المدرسة", "ar", "paraphrase", ("007",)),
    EvalCase("تسوون فحوصات للعمالة؟", "ar", "clean", ("007", "017")),
    EvalCase("ليش أختاركم انتم بالذات؟", "ar", "clean", ("008",)),
    EvalCase("وش يميزكم عن باقي المراكز؟", "ar", "clean", ("008",)),
    EvalCase("وش الفايدة لي إني أستخدم موفق؟", "ar", "ambiguous", ("008", "001", "010")),
    EvalCase("كيف أحجز فحص؟", "ar", "clean", ("009",)),
    EvalCase("أبغى أحجز موعد، كيف الطريقة؟", "ar", "clean", ("009",)),
    EvalCase("وش خطوات الحجز عندكم؟", "ar", "clean", ("009",)),
    EvalCase("بعد ما أحجز وش أسوي؟", "ar", "paraphrase", ("009",)),
    EvalCase("كيف تجيني النتيجة بعد ما أخلص الفحص؟", "ar", "paraphrase", ("009", "010")),

    # ================= التطبيق  (FAQ 010) =================
    EvalCase("فيه تطبيق جوال لموفق؟", "ar", "clean", ("010",)),
    EvalCase("وش يسوي التطبيق حقكم؟", "ar", "clean", ("010",)),
    EvalCase("أقدر أتابع فحصي من الجوال؟", "ar", "clean", ("010",)),
    EvalCase("ينزل التقرير على جوالي؟", "ar", "clean", ("010",)),
    EvalCase("فيه تنبيهات تذكرني بالمواعيد؟", "ar", "clean", ("010",)),

    # ================= الدفع  (FAQ 011) =================
    EvalCase("كيف أقدر أدفع؟", "ar", "clean", ("011",)),
    EvalCase("وش طرق الدفع عندكم؟", "ar", "clean", ("011",)),
    EvalCase("تقبلون تابي؟", "ar", "clean", ("011",)),
    EvalCase("تمارا موجودة عندكم؟", "ar", "clean", ("011",)),
    EvalCase("أقدر أدفع بآبل باي؟", "ar", "clean", ("011",)),
    EvalCase("مدى يشتغل معكم؟", "ar", "clean", ("011",)),
    EvalCase(
        "أقدر أدفع كاش في المركز؟", "ar", "ambiguous", ("011",),
        note=(
            "The corpus lists digital methods only and never mentions cash, so "
            "the honest reply hedges. Only retrieval of FAQ 011 is asserted."
        ),
    ),

    # ================= الأكاديمية  (FAQ 012-014) =================
    EvalCase("وش هي أكاديمية موفق؟", "ar", "clean", ("012",)),
    EvalCase("الأكاديمية تدرب على ايش؟", "ar", "clean", ("012",)),
    EvalCase("وش مميزات التدريب عندكم؟", "ar", "clean", ("013",)),
    EvalCase("الدورات مسجلة ولا مباشر؟", "ar", "clean", ("013",)),
    EvalCase("فيه اختبارات تجريبية قبل الاختبار الحقيقي؟", "ar", "clean", ("013",)),
    EvalCase("التدريب عندكم بكم لغة؟", "ar", "clean", ("014", "013")),
    EvalCase("كم نسبة النجاح عندكم؟", "ar", "clean", ("014",)),
    EvalCase("فيه لوحة تحكم أتابع فيها تدريبي؟", "ar", "clean", ("013",)),
    EvalCase("الأسعار موحدة في الأكاديمية؟", "ar", "clean", ("013",)),

    # ================= موفق أعمال  (FAQ 015-021) =================
    EvalCase("وش هو موفق أعمال؟", "ar", "clean", ("015",)),
    EvalCase(
        "عندي شركة، وش تقدمون لي؟", "ar", "paraphrase", ("015", "016", "017"),
        note="Business intent with no product name mentioned at all.",
    ),
    EvalCase("ليش الشركات تحتاج موفق أعمال؟", "ar", "clean", ("016",)),
    EvalCase("وش الفايدة اللي بتعود على شركتي؟", "ar", "paraphrase", ("016",)),
    EvalCase(
        "تختصرون علينا ملفات الإكسل والمتابعة اليدوية؟", "ar", "paraphrase", ("016",),
        note="Quotes a pain point stated only inside FAQ 016's answer text.",
    ),
    EvalCase("فيه تنبيهات لانتهاء الرخص والشهادات؟", "ar", "clean", ("016", "017")),
    EvalCase("وش خدمة الفحوصات اللي تقدمونها للشركات؟", "ar", "clean", ("017",)),
    EvalCase("تجدولون فحوصات الموظفين؟", "ar", "clean", ("017",)),
    EvalCase("فيه ربط مباشر مع المراكز الطبية؟", "ar", "clean", ("017", "019")),
    EvalCase("فيه تدريب لموظفيني؟", "ar", "clean", ("018",)),
    EvalCase("تدربون الكوادر عندكم؟", "ar", "clean", ("018",)),
    EvalCase("وش هي خدمة الصحة المهنية عندكم؟", "ar", "clean", ("019",)),
    EvalCase("تسوون فحوصات بيئية للمنشأة؟", "ar", "clean", ("019",)),
    EvalCase("وش مراحل التسجيل للشركات عندكم؟", "ar", "clean", ("020",)),
    EvalCase("كيف تتحققون من السجل التجاري؟", "ar", "clean", ("020",)),
    EvalCase("وش خطوات إدارة فحوصات الفريق؟", "ar", "clean", ("021",)),
    EvalCase("كيف أدير فحوصات موظفيني؟", "ar", "clean", ("021", "017")),
    EvalCase("فيه لوحة تحكم موحدة أشوف فيها كل الموظفين؟", "ar", "ambiguous",
             ("016", "017", "019", "021")),
    EvalCase("أبغى أتابع امتثال شركتي، تقدرون؟", "ar", "paraphrase", ("016", "021", "019")),

    # ================= عام والجمهور  (FAQ 022) =================
    EvalCase("مين اللي يستخدم موفق؟", "ar", "clean", ("022",)),
    EvalCase("موفق للأفراد ولا للشركات؟", "ar", "clean", ("022",)),
    EvalCase("وش الفئات اللي تخدمونها؟", "ar", "clean", ("022",)),

    # ================= مسيرة الشركة  (FAQ 023) =================
    EvalCase("متى تأسست موفق؟", "ar", "clean", ("023",)),
    EvalCase("وش صار عندكم في 2024؟", "ar", "clean", ("023",)),
    EvalCase("وش أهم محطاتكم من البداية؟", "ar", "clean", ("023",)),

    # ================= التواصل  (FAQ 024) =================
    EvalCase("وين مقركم؟", "ar", "clean", ("024",)),
    EvalCase("مقر الشركة في أي مدينة؟", "ar", "clean", ("024",)),
    EvalCase("كيف أتواصل معكم؟", "ar", "clean", ("024",)),
    EvalCase("أبغى أكلم مسؤول نجاح العملاء", "ar", "ambiguous", ("024", "026")),

    # ================= البداية  (FAQ 025-026) =================
    EvalCase("أبغى أبدأ معكم كفرد، كيف؟", "ar", "clean", ("025",)),
    EvalCase("أول خطوة عشان أستخدم موفق وش هي؟", "ar", "ambiguous", ("025", "009")),
    EvalCase("شركتي كيف تبدأ معكم؟", "ar", "clean", ("026",)),
    EvalCase("أبغى أفتح حساب شركة", "ar", "clean", ("026",)),
    EvalCase("فيه عرض تجريبي للشركات؟", "ar", "clean", ("026",)),

    # ======= تفاصيل غير محددة في المصدر: ممنوع الاختراع  (FAQ 024) =======
    EvalCase(
        "كم رقم جوالكم؟", "ar", "no_fabrication", ("024",),
        forbidden_pattern=CONTACT_PATTERN,
        note="Phone was left unfinalised in the source; a number must never appear.",
    ),
    EvalCase(
        "وش الإيميل حقكم؟", "ar", "no_fabrication", ("024",),
        forbidden_pattern=CONTACT_PATTERN,
    ),
    EvalCase(
        "عطني رقم التواصل المباشر", "ar", "no_fabrication", ("024",),
        forbidden_pattern=CONTACT_PATTERN,
        note="Imperative phrasing pressures the model harder than a question.",
    ),

    # ================= خارج النطاق: لازم يعتذر =================
    EvalCase("بكم الفحص الطبي؟", "ar", "out_of_scope",
             note="No pricing anywhere in the corpus."),
    EvalCase("كم سعر فحص الإقامة بالريال؟", "ar", "out_of_scope"),
    EvalCase("أقدر ألغي موعدي؟", "ar", "out_of_scope",
             note="No cancellation policy in the corpus."),
    EvalCase("فيه استرجاع فلوس لو ما رحت للموعد؟", "ar", "out_of_scope"),
    EvalCase("وش دوامكم؟ متى تفتحون؟", "ar", "out_of_scope",
             note="No opening hours in the corpus."),
    EvalCase("وش عنوان فرعكم في الدمام؟", "ar", "out_of_scope",
             note="Only the Riyadh head office is documented; no branch list."),
    EvalCase("تشتغلون في الكويت والإمارات؟", "ar", "out_of_scope",
             note="Corpus covers Saudi Arabia only; must not extrapolate."),
    EvalCase("كم مدة صلاحية الشهادة الصحية؟", "ar", "out_of_scope"),
    EvalCase("فيه خصم للطلاب؟", "ar", "out_of_scope"),
    EvalCase("كم تكلفة اشتراك الشركات؟", "ar", "out_of_scope"),
    EvalCase("عندكم خدمة فحص منزلي؟", "ar", "out_of_scope"),
    EvalCase("كم عدد موظفين شركة موفق؟", "ar", "out_of_scope"),
)
