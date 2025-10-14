Цель

Собрать, фильтровать и анализировать научные статьи по запросу «aging theory» (и производным), автоматически порождать и уточнять онтологию теорий старения, расширять корпус публикаций через итеративную генерацию запросов и визуализировать карту знаний. Решение должно быть:
- воспроизводимым (одна команда запуска, зафиксированные версии, детерминированные параметры);
- масштабируемым (инкрементальные обновления, модульная архитектура);
- понятным экспертам (прозрачные отчёты, объяснимые метки/решения LLM).

Право и этика (обязательно)
Запрещено: использование статей где не раскрывается никаких теорий старения, где рассуждается только о конкретных механизмах, либо рассуждается только о старении мозга или каких-либо частей организма.
Разрешено: sci-hub API, openalex API открытый доступ (OA) и API: OpenAlex, PubMed/PMC (NCBI E‑utilities), Crossref, Semantic Scholar API, Unpaywall, CORE, OpenAIRE, arXiv/bioRxiv/medRxiv, издательские страницы с явной OA‑лицензией.
Google Scholar: при необходимости — опираться на законные промежуточные сервисы (напр., SerpAPI c соблюдением их ToS) или, предпочтительно, на OpenAlex/Unpaywall/Semantic Scholar.

Результаты
Корпус: метаданные + аннотации + (где разрешено) full text.
Онтология теорий старения (узлы/подузлы/отношения) + версии (semver).
Граф: документы ↔ теории (поддерживает/опровергает/обсуждает), связи между теориями.
Итеративная воронка поиска: автоматически сгенерированные запросы для расширения корпуса.
Отчёты: метрики покрытия/точности, журналы решений LLM, визуальная карта.

Архитектура модулей
ingest/             # загрузка метаданных из API
oa_resolver/        # поиск легальных full-text (Unpaywall/PMC/CORE/OpenAIRE)
text_parse/         # HTML/PDF→структура (GROBID/Tesseract при необходимости)
filtering/          # релевантность к "aging theory" (классификатор + LLM)
ontology_induction/ # выделение кандидатов теорий из обзоров
graph_refine/       # иерархизация/расщепление/слияние узлов
query_gen/          # генерация поисковых фраз для следующей итерации
retrieval/          # дозагрузка новых статей, дедуп, обновление графа
eval_reporting/     # метрики, отчёты, визуализация
orchestrator/       # пайплайн (CLI), контроль версий, логи
storage/            # БД документов + граф онтологии


Стек (референс):
API: OpenAlex, sci-hub, PubMed/PMC (E‑utilities), Crossref, Semantic Scholar, Unpaywall, CORE, OpenAIRE.
Векторный поиск: FAISS/LanceDB; эмбеддинги: sentence-transformers (научные модели, напр. allenai/scibert_scivocab для токенизации, sentence-transformers/all-mpnet-base-v2 или научные ST).
Классификация/переранжирование: cross-encoder/ms-marco-MiniLM-L-6-v2 или подобный.
Парсинг: GROBID (PDF→TEI XML→Plain), Tesseract (OCR при необходимости).
NER/термин-майнинг: spaCy (scispaCy), YAKE/KeyBERT/SGRank.
Тематическое моделирование: BERTopic (UMAP+HDBSCAN), альтернативно LDA (базовое).
Граф: Neo4j / ArangoDB / NetworkX + экспорт в JSON-LD.
Оркестрация: Prefect / Airflow (на хакатоне можно make + python -m).
Визуализация: Streamlit/Gradio + Cytoscape.js для графа.

Схемы данных
Документ
{
  "id": "doi:10.1234/abcd",
  "source_ids": ["openalex:W...", "pubmed:12345"],
  "title": "...",
  "abstract": "...",
  "year": 2021,
  "venue": "Journal ...",
  "authors": [{"name":"...", "orcid":"..."}],
  "license": "CC-BY-4.0",
  "oa_status": "gold|green|hybrid|bronze|closed",
  "fulltext_url": "https://...",
  "fulltext_path": "s3://.../doi_10_1234_abcd.pdf",
  "text_checksum": "sha256:...",
  "entities": {
    "organisms": ["Homo sapiens","Mus musculus"],
    "interventions": ["rapamycin","CR","telomere extension"],
    "outcomes": ["lifespan","healthspan"],
    "theory_mentions": ["antagonistic pleiotropy","disposable soma", "..."]
  },
  "relevance_score": 0.0,   // к темам теорий старения
  "is_review": true,
  "notes": "provenance/logs..."
}

Онтология (фрагмент)
{
  "theory_id": "theory:antagonistic_pleiotropy",
  "label": "Antagonistic pleiotropy",
  "synonyms": ["АП-гипотеза", "antagonistic pleiotropy theory"],
  "parents": ["theory:evolutionary"],
  "children": ["theory:age_specific_effects"],
  "description": "краткая выжимка из обзоров",
  "doc_links": [
    {"doc_id":"doi:10.1234/..","relation":"supports|refutes|discusses","evidence_type":"empirical|theoretical|meta-analysis"}
  ],
  "stats": {"n_docs": 37, "n_supports": 22, "n_refutes": 4}
}

Пайплайн (соответствует вашим шагам 1–6)
Шаг 1. Поисковый сбор и пред-фильтрация (обзоры)
Вход: базовая фраза "aging theory" (+ морфологии: "ageing theory", "theories of aging", "senescence theory").
Действия:
ingest/:
Запросы к OpenAlex / PubMed / Sci-Hub Semantic Scholar с фильтрами:
язык: en и ru/, тип публикации: review/meta-analysis/book-chapter (опционально), годы: не ограничивать.
Сохранять: title, abstract, year, type, doi, concepts, referenced_works, citations_count.
filtering/:
Классификатор релевантности (двухступенчатый):
Heuristic prefilter: наличие токенов aging|ageing|senescence + theor*|model|hypothesis в title+abstract.
LLM/ML классификация is_about_aging_theory ∈ {true,false,uncertain} + объяснение (лог).
Отбросить нерелевантные записи.
Выход: corpus_seed.jsonl (обзоры, релевантные теории старения).
Подсказка LLM (пример):
Реши, относится ли статья к теоретическим моделям старения (а не к частным биомаркерам/клиническим случаям). Верни JSON: {decision, confidence, rationale, key_terms}.

Шаг 2. Легальный фуллтекст и парсинг
Цель: получить текст только при наличии законной лицензии.
oa_resolver/:
Для каждого DOI обратиться к Unpaywall: OA‑статус и ссылка на полные тексты.
Альтернативы/дополнения: PubMed Central, CORE, OpenAIRE, страницы издателей (если отображают OA‑лицензию).
text_parse/:
Скачать OA PDF/HTML.
GROBID → TEI → структурированный текст (секции, абстракт, заголовки, таблицы/фигуры‑плейсхолдеры).
OCR при необходимости (Tesseract).
Нормализация, токенизация, снятие стоп‑слов (без агрессивной стемминга для терминов).
Для записей без OA — храним только метаданные и аннотацию.
Выход: docs_store/ (тексты+метаданные), чек‑листы лицензий.

Шаг 3. Индукция онтологии «с нуля» из обзоров
Идея: не задаём заранее перечень теорий; извлекаем кандидатов из обзоров.
ontology_induction/:
Детектируем обзоры (is_review==true) → повышенный вес.
Термин‑майнинг: KeyBERT/YAKE по секциям Introduction/Background.
NER (scispaCy) + шаблоны «X theory/hypothesis/model of aging».
Слияние синонимов (эмбеддинги + лексические правила).
Кандидат в «теорию» принимается при выполнении минимум двух критериев:
упоминания ≥ m в независимых обзорах (напр., m=2);
контекстные индикаторы теоретичности (слова theory, hypothesis, model, framework рядом).
Первичный граф: узлы‑теории, рёбра related_to по схожести контекста (cosine ≥ τ).
Выход: ontology_v0.json (черновик).

Шаг 4. Перестройка графа (слияние/расщепление) по “оптимальному размеру”
Задача: если в узле копится слишком много статей, делим; если обнаружена «чужая» теория — перенос/создание нового узла.
Размер узла‑по‑умолчанию: 10–40 документов.
Правила:
Если n_docs > 40:
Документы узла → эмбеддинги → HDBSCAN/агломеративная кластеризация.
Если найдены устойчивые подкластер(а) (silhouette ≥ 0.2, размер ≥ 8), создаём дочерние узлы; родитель становится группой теорий.
Если документ по контенту ближе к другому узлу (межузловое сходство > 0.6) — перенос.
Если кластер обособлен от всех (макс‑сходство < 0.35) — новая независимая теория.
Маршрут решений логируется; в графе фиксируются связи parent_of, derived_from, overlaps_with.
Выход: ontology_v1.json + graph_log.md.

Шаг 5. Генерация поисковых фраз и дозагрузка НЕ только обзоров
Цель: расширение корпуса эмпирическими/методологическими работами.
query_gen/:
Для каждого узла: сгенерировать 10–20 поисковых запросов трёх типов:
Boolean‑строки для API (напр.: (“antagonistic pleiotropy” OR “age-specific trade-off”) AND (aging OR ageing OR senescence)).
Словари синонимов/морфологии (американское/британское написание).
Негативные фильтры (исключать омонимные домены, напр. «battery aging» для материаловедения, если не нужно).
Включить термины организмов/моделей, если узел об этом (e.g., Drosophila, C. elegans, mice).
retrieval/:
Запросы к OpenAlex/PubMed/Semantic Scholar/Crossref.
Дедуп по DOI/заголовку (MinHash/SimHash на нормализованном названии).

Повторить Шаг 2 (полные тексты — только OA) и Шаг 3–4 (обогащение онтологии).
Выход: обновлённый docs_store/, ontology_v{n+1}.json.

Шаг 6. Итеративный цикл (1–5)
Останавливаемся при выполнении критериев качества (см. ниже) либо при стабилизации онтологии (≤5% изменений узлов за итерацию).
Каждая итерация — новый release онтологии (semver: 0.x.y), артефакты в releases/.

Метрики качества (для «победной» презентации)
Покрытие: доля обзоров, в которых обнаружены узлы онтологии (≥80% целевых обзоров).
Сжатость: средний размер узла в пределах 10–40; дисперсия не выше заданного порога.
Семантическая чистота: внутр. сходство документов узла − межузловое сходство ≥ Δ (напр., ≥0.15).
Объяснимость: для каждого узла — краткая выжимка (3–5 предложений) из обзоров + топ‑цитируемые источники.
Доказательность: статистика supports/refutes/discusses по каждому узлу (на уровне предложений/пассов).
Репродуцируемость: make all приводит к тем же показателям на контрольном сэмпле.

Компоненты LLM (промпты и IO‑контракты)
A. Релевантность к теории старения
Вход: title, abstract.
Выход (JSON):
{"decision":"true|false|uncertain","confidence":0.0,"rationale":"...", "key_terms":["...","..."]}

B. Экстракция кандидатов‑теорий из обзоров
Вход: структурированный текст обзора (весомее секции Intro/Discussion).
Задача: выделить названия теорий/гипотез/моделей старения, их синонимы, краткое описание, ключевые термины, класс (эволюционные/стохастические/поведенческие/клеточные/информационные и т. д. — если явный сигнал).
Выход (JSON):
[{
  "label":"<theory_name>",
  "synonyms":["..."],
  "snippet":"краткая выжимка (1-2 предложения)",
  "evidence":["<цитата/ссылка секции>"],
  "confidence":0.0
}]

C. Маркеры отношения документа к узлу
Вход: документ (абстракт или параграфы результатов) + описание узла.
Выход (JSON):
{"relation":"supports|refutes|discusses|unclear","evidence_spans":[{"text":"...","section":"Results"}], "confidence":0.0}

D. Генерация поисковых фраз
Вход: узел онтологии (label, синонимы, ключевые термины, организмы, типы доказательств).
Выход (JSON):

{
  "boolean_queries":[ "...", "..." ],
  "positive_terms":["..."],
  "negative_terms":["battery aging","material degradation"],
  "organism_terms":["Homo sapiens","Drosophila"],
  "method_terms":["meta-analysis","GWAS","systematic review"]
}

Визуализация
Веб‑интерфейс: карта теорий (Cytoscape.js), размер узла ~ n_docs, цвет ~ баланс supports/refutes.
Клик по узлу → карточка: описание, топ‑источники (OA ссылки), ключевые фразы, дочерние узлы.
Поиск по теории/синонимам/организмам.
Псевдокод основного цикла
def main():
    seed_queries = ["aging theory", "ageing theory", "theories of aging", "senescence theory"]

    # 1) Ingest seed reviews
    reviews = ingest_from_apis(seed_queries, kind="review")
    reviews = prefilter_heuristics(reviews)
    reviews = llm_filter(reviews)  # decision/confidence/logs

    # 2) Resolve OA + parse
    docs = resolve_oa_and_parse(reviews)  # Unpaywall/PMC/CORE/OpenAIRE + GROBID

    # 3) Induce ontology from reviews
    ontology = induce_ontology(docs)

    # 4) Refine graph (split/merge/move)
    ontology = refine_graph(ontology, docs)

    # 5) Generate queries per node & retrieve new docs
    node_queries = llm_generate_queries(ontology)
    new_docs = ingest_from_apis(node_queries, kind="all")
    new_docs = deduplicate(new_docs)
    new_docs = resolve_oa_and_parse(new_docs)

    # 6) Update ontology, loop until convergence
    ontology = update_ontology(ontology, new_docs)
    if not converged(ontology):
        repeat from step 4

Критические эвристики и параметры (по умолчанию)
m (минимум упоминаний теории в разных обзорах): 2.
Порог cosine для related_to: 0.45.
Критерий расщепления узла: n_docs>40, silhouette ≥ 0.2, размер подкластеров ≥ 8.
Межузловое сходство для переноса: >0.6; для новой независимой теории: <0.35 к ближайшему.
Размер итогового узла‑цели для показа: 10–40 документов.

Хранилище и версияция
Документы: Parquet + контент хеша (sha256), отдельная таблица licenses.
Векторные индексы: FAISS per‑node + общий глобальный.
Онтология: ontology_vN.json (+ JSON‑LD экспорт).
Полные логи LLM‑решений: logs/decisions.jsonl.

Все артефакты фиксируются в releases/ (semver), сборка через make.

Проверка качества и демо‑артефакты

Тест‑сэмпл из 100 статей (ручная разметка релевантности и отношения supports/refutes/discusses).

Отчёт report.md: метрики, графики распределений, примеры «объяснимых» спанов текста.

Демо‑веб: поиск узлов, раскрытие подузлов, скачиваемые списки статей (CSV/JSON) с OA‑ссылками.

Команды CLI (референс)
make bootstrap        # env, ключи API, GROBID docker up
make seed_ingest      # Шаг 1
make resolve_parse    # Шаг 2
make induce           # Шаг 3
make refine           # Шаг 4
make expand_search    # Шаг 5
make iterate          # Шаг 6 (несколько проходов)
make viz              # локальная визуализация
make report           # метрики и отчёт


ENV-переменные: OPENALEX_API_KEY (если есть), NCBI_EMAIL, PUBMED_KEY, SEMANTIC_SCHOLAR_KEY, UNPAYWALL_EMAIL, CORE_API_KEY, OPENAIRE_KEY.

Риски и смягчения
Смещение LLM: хранить рационали; пороги принятия решений; ручной sanity‑чек на сэмпле.
Шумы доменов (battery aging и т. п.): негативные термины и фильтры предметных категорий.
PDF‑качество: GROBID+OCR; оценка доли распознанного текста; fallback к аннотациям.

Критерии готовности (Go/No-Go)
Онтология покрывает ≥80% терминов, встречающихся в ≥2 обзорах.
Не менее 10 основных теорий с краткими описаниями и списками ключевых работ.
Веб‑демо позволяет навигацию по узлам/подузлам; присутствуют цифры supports/refutes.

Отчёт с метриками и повторяемой сборкой (make all) воспроизводит результаты на тестовом сэмпле.
