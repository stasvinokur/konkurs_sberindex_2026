# Лицензии зависимостей

Сформировано командой `uv run --with pip-licenses pip-licenses --format=markdown --with-urls` 2026-10-04; пакетов: 118.

Лицензии **весов моделей** — отдельный документ [THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md):
лицензия кода библиотеки и лицензия её весов различаются, и именно на весах встречаются
некоммерческие условия. Собственный код решения — под MIT ([LICENSE](LICENSE)).

## Итог проверки

- Пакетов с некоммерческими или проприетарными платными лицензиями: **0**.
- Пакетов под GPL-семейством: **1**.

### Единственная копилефт-зависимость

**fpdf2 2.8.8** — LGPL-3.0-only (https://py-pdf.github.io/fpdf2/).

`fpdf2` используется только для экспорта методологического отчёта в PDF. LGPL разрешает
использование библиотеки в работе под другой лицензией при условии, что пользователь может
её заменить: она подключается как обычная зависимость из PyPI, не модифицируется и не
встраивается в код решения, поэтому требований раскрывать собственный код не возникает.
Если условие окажется неприемлемым, экспорт заменяется на `reportlab` (BSD-3-Clause);
затронут будет один модуль `src/sbx/shell/render/pdf.py`.

## Шрифты лендинга

Лендинг `reports/landing/index.html` набран шрифтами IBM Plex; они встроены в сам файл, чтобы
страница открывалась без сети. В репозитории лежат официальные неизменённые файлы-подмножества
(`src/sbx/shell/render/assets/fonts/`, 14 файлов WOFF2, 257 КБ) и текст лицензии рядом с ними
(`LICENSE.txt`). Лицензия названа и в самой странице — в начале встроенной таблицы стилей.

| Шрифт | Начертания | Пакет npm, версия | Лицензия |
|---|---|---|---|
| IBM Plex Sans | 400, 500, 600 | `@ibm/plex-sans` 1.1.0 | SIL Open Font License 1.1 |
| IBM Plex Sans Condensed | 600 | `@ibm/plex-sans-condensed` 2.0.0 | SIL Open Font License 1.1 |
| IBM Plex Mono | 400 | `@ibm/plex-mono` 2.5.0 | SIL Open Font License 1.1 |

Правообладатель — IBM Corp. (Copyright © 2017 IBM Corp. with Reserved Font Name "Plex"),
исходный репозиторий — https://github.com/IBM/plex. Из каждого пакета взяты файлы
`fonts/split/woff2/*-{Cyrillic,Latin1,Latin2,Pi}.woff2`: кириллица, латиница, знак рубля,
стрелки и математические знаки. OFL разрешает встраивать и распространять шрифты вместе с
программой и документами; файлы не изменялись, поэтому условие о зарезервированном имени не
затронуто. Код решения остаётся под MIT, на шрифты эта лицензия не распространяется.

## Распределение по лицензиям

| Лицензия | Пакетов |
|---|---|
| MIT | 32 |
| Apache Software License | 17 |
| BSD-3-Clause | 14 |
| BSD License | 13 |
| MIT License | 10 |
| Apache-2.0 | 9 |
| Python Software Foundation License | 3 |
| BSD-2-Clause | 2 |
| Apache-2.0 AND MIT | 1 |
| Mozilla Public License 2.0 (MPL 2.0) | 1 |
| LGPL-3.0-only | 1 |
| MPL-2.0 | 1 |
| BSD-2-Clause AND Apache-2.0 WITH LLVM-exception | 1 |
| Apache License 2.0 | 1 |
| BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | 1 |
| Apache-2.0 OR BSD-2-Clause | 1 |
| MIT-CMU | 1 |
| 3-Clause BSD License | 1 |
| Apache Software License; BSD License | 1 |
| Apache 2.0 | 1 |
| Apache-2.0 AND CNRI-Python | 1 |
| ISC License (ISCL) | 1 |
| Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | 1 |
| MPL-2.0 AND MIT | 1 |
| Apache 2.0 License | 1 |
| PSF-2.0 | 1 |

## Полный список

| Пакет | Версия | Лицензия | Источник |
|---|---|---|---|
| accelerate | 1.15.0 | Apache Software License | https://github.com/huggingface/accelerate |
| adagio | 0.2.6 | Apache Software License | http://github.com/fugue-project/adagio |
| aiohappyeyeballs | 2.7.1 | Python Software Foundation License | https://github.com/aio-libs/aiohappyeyeballs |
| aiohttp | 3.14.3 | Apache-2.0 AND MIT | https://github.com/aio-libs/aiohttp |
| aiosignal | 1.4.0 | Apache Software License | https://github.com/aio-libs/aiosignal |
| alembic | 1.20.0 | MIT | https://alembic.sqlalchemy.org |
| annotated-doc | 0.0.5 | MIT | https://github.com/fastapi/annotated-doc |
| annotated-types | 0.8.0 | MIT | https://github.com/annotated-types/annotated-types |
| anyio | 4.15.1 | MIT | https://anyio.readthedocs.io/en/stable/versionhistory.html |
| attrs | 26.1.0 | MIT | https://www.attrs.org/en/stable/changelog.html |
| certifi | 2026.7.22 | Mozilla Public License 2.0 (MPL 2.0) | https://github.com/certifi/python-certifi |
| charset-normalizer | 3.5.1 | MIT | https://github.com/jawah/charset_normalizer/blob/master/CHANGELOG.md |
| chronos-forecasting | 2.3.2 | Apache Software License | https://github.com/amazon-science/chronos-forecasting |
| click | 8.5.0 | BSD-3-Clause | https://github.com/pallets/click/ |
| cloudpickle | 3.1.2 | BSD License | https://github.com/cloudpipe/cloudpickle |
| cmdstanpy | 1.3.0 | BSD License | https://github.com/stan-dev/cmdstanpy |
| colorlog | 6.12.0 | MIT License | https://github.com/borntyping/python-colorlog |
| contourpy | 1.4.0 | BSD-3-Clause | https://github.com/contourpy/contourpy |
| coreforecast | 0.0.18 | Apache Software License | https://nixtla.github.io/coreforecast |
| cycler | 0.12.1 | BSD License | https://matplotlib.org/cycler/ |
| defusedxml | 0.7.1 | Python Software Foundation License | https://github.com/tiran/defusedxml |
| einops | 0.8.2 | MIT License | https://github.com/arogozhnikov/einops |
| filelock | 3.32.6 | MIT | https://github.com/tox-dev/py-filelock |
| fonttools | 4.65.0 | MIT | http://github.com/fonttools/fonttools |
| formulaic | 1.2.2 | MIT | https://github.com/matthewwardrop/formulaic |
| fpdf2 | 2.8.8 | LGPL-3.0-only | https://py-pdf.github.io/fpdf2/ |
| frozenlist | 1.8.0 | Apache-2.0 | https://github.com/aio-libs/frozenlist |
| fsspec | 2026.7.0 | BSD-3-Clause | https://github.com/fsspec/filesystem_spec |
| fugue | 0.9.4 | Apache Software License | http://github.com/fugue-project/fugue |
| h11 | 0.16.0 | MIT License | https://github.com/python-hyper/h11 |
| hf-xet | 1.6.0 | Apache-2.0 | https://github.com/huggingface/xet-core |
| holidays | 0.104 | MIT | https://github.com/vacanza/holidays/ |
| httpcore | 1.0.9 | BSD-3-Clause | https://www.encode.io/httpcore/ |
| httpx | 0.28.1 | BSD License | https://github.com/encode/httpx |
| huggingface_hub | 1.31.0 | Apache Software License | https://github.com/huggingface/huggingface_hub |
| hypothesis | 6.168.0 | MPL-2.0 | https://hypothesis.works |
| idna | 3.19 | BSD-3-Clause | https://github.com/kjd/idna |
| iniconfig | 2.3.0 | MIT | https://github.com/pytest-dev/iniconfig |
| interface_meta | 2.0.1 | MIT | https://github.com/matthewwardrop/interface_meta |
| Jinja2 | 3.1.6 | BSD License | https://github.com/pallets/jinja/ |
| joblib | 1.6.0 | BSD-3-Clause | https://joblib.readthedocs.io |
| jsonschema | 4.26.0 | MIT | https://github.com/python-jsonschema/jsonschema |
| jsonschema-specifications | 2025.9.1 | MIT | https://github.com/python-jsonschema/jsonschema-specifications |
| kiwisolver | 1.5.1 | BSD License | https://github.com/nucleic/kiwi |
| lightgbm | 4.7.0 | MIT | https://github.com/lightgbm-org/LightGBM |
| lightning-utilities | 0.15.3 | Apache-2.0 | https://github.com/Lightning-AI/utilities |
| llvmlite | 0.49.0 | BSD-2-Clause AND Apache-2.0 WITH LLVM-exception | http://llvmlite.readthedocs.io |
| Mako | 1.4.1 | MIT | https://www.makotemplates.org/ |
| markdown-it-py | 4.2.0 | MIT License | https://github.com/executablebooks/markdown-it-py |
| MarkupSafe | 3.0.3 | BSD-3-Clause | https://github.com/pallets/markupsafe/ |
| matplotlib | 3.11.2 | Python Software Foundation License | https://matplotlib.org |
| mdurl | 0.1.2 | MIT License | https://github.com/executablebooks/mdurl |
| mpmath | 1.3.0 | BSD License | http://mpmath.org/ |
| msgpack | 1.2.2 | Apache-2.0 | https://msgpack.org/ |
| multidict | 6.8.0 | Apache License 2.0 | https://github.com/aio-libs/multidict |
| narwhals | 2.26.0 | MIT | https://github.com/narwhals-dev/narwhals |
| networkx | 3.6.1 | BSD-3-Clause | https://networkx.org/ |
| neuralforecast | 3.1.7 | Apache Software License | https://github.com/Nixtla/neuralforecast/ |
| numba | 0.67.0 | BSD License | https://numba.pydata.org |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://numpy.org |
| optuna | 5.0.0 | MIT License | https://optuna.org/ |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging |
| pandas | 3.0.5 | BSD License | https://pandas.pydata.org |
| patsy | 1.0.3 | BSD License | https://github.com/pydata/patsy |
| pillow | 12.3.0 | MIT-CMU | https://python-pillow.github.io |
| pluggy | 1.6.0 | MIT License |  |
| propcache | 0.5.3 | Apache-2.0 | https://github.com/aio-libs/propcache |
| prophet | 1.4.0 | MIT | https://facebook.github.io/prophet/ |
| protobuf | 7.36.1 | 3-Clause BSD License | https://developers.google.com/protocol-buffers/ |
| psutil | 7.2.2 | BSD-3-Clause | https://github.com/giampaolo/psutil |
| pyarrow | 25.0.1 | Apache-2.0 | https://arrow.apache.org/ |
| pydantic | 2.13.5 | MIT | https://github.com/pydantic/pydantic |
| pydantic_core | 2.46.5 | MIT | https://github.com/pydantic |
| Pygments | 2.21.0 | BSD-2-Clause | https://pygments.org |
| pyparsing | 3.3.2 | MIT | https://github.com/pyparsing/pyparsing/ |
| pytest | 9.1.1 | MIT | https://docs.pytest.org/en/latest/ |
| python-dateutil | 2.9.0.post0 | Apache Software License; BSD License | https://github.com/dateutil/dateutil |
| pytorch-lightning | 2.6.6 | Apache Software License | https://github.com/Lightning-AI/lightning |
| PyYAML | 6.0.3 | MIT License | https://pyyaml.org/ |
| ray | 2.58.0 | Apache 2.0 | https://github.com/ray-project/ray |
| referencing | 0.37.0 | MIT | https://github.com/python-jsonschema/referencing |
| regex | 2026.9.10 | Apache-2.0 AND CNRI-Python | https://github.com/mrabarnett/mrab-regex |
| requests | 2.34.2 | Apache Software License | https://github.com/psf/requests |
| rich | 15.0.0 | MIT License | https://github.com/Textualize/rich |
| river | 0.26.1 | BSD-3-Clause | https://riverml.xyz/ |
| rpds-py | 2026.6.3 | MIT | https://github.com/crate-py/rpds |
| ruff | 0.16.7 | MIT | https://docs.astral.sh/ruff |
| ruptures | 1.1.10 | BSD License | https://github.com/deepcharles/ruptures/ |
| safetensors | 0.8.0 | Apache Software License | https://github.com/huggingface/safetensors |
| sbx | 0.1.0 | MIT |  |
| scikit-learn | 1.9.1 | BSD-3-Clause | https://scikit-learn.org |
| scipy | 1.18.1 | BSD License | https://scipy.org/ |
| shellingham | 1.5.4 | ISC License (ISCL) | https://github.com/sarugaku/shellingham |
| six | 1.17.0 | MIT License | https://github.com/benjaminp/six |
| sortedcontainers | 2.4.0 | Apache Software License | http://www.grantjenks.com/docs/sortedcontainers/ |
| SQLAlchemy | 2.0.54 | MIT | https://www.sqlalchemy.org |
| stanio | 0.5.1 | BSD-3-Clause | https://github.com/stan-dev/stanio |
| statsforecast | 2.0.1 | Apache Software License | https://github.com/Nixtla/statsforecast/ |
| statsmodels | 0.15.0 | BSD-3-Clause | https://www.statsmodels.org |
| sympy | 1.14.0 | BSD License | https://sympy.org |
| tabulate | 0.10.0 | MIT | https://github.com/astanin/python-tabulate |
| tensorboardX | 2.6.5 | MIT | https://github.com/lanpa/tensorboardX |
| threadpoolctl | 3.7.0 | BSD-3-Clause | https://github.com/joblib/threadpoolctl |
| timesfm | 3.0.2 | Apache-2.0 |  |
| tokenizers | 0.23.2 | Apache Software License | https://github.com/huggingface/tokenizers |
| torch | 2.14.0 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | https://pytorch.org |
| torchmetrics | 1.9.0 | Apache Software License | https://github.com/Lightning-AI/torchmetrics |
| tornado | 6.5.10 | Apache Software License | http://www.tornadoweb.org/ |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | https://tqdm.github.io |
| transformers | 5.17.0 | Apache 2.0 License | https://github.com/huggingface/transformers |
| triad | 1.0.2 | Apache-2.0 | http://github.com/fugue-project/triad |
| typer | 0.27.2 | MIT | https://github.com/fastapi/typer |
| typing-inspection | 0.4.4 | MIT | https://github.com/pydantic/typing-inspection |
| typing_extensions | 4.16.0 | PSF-2.0 | https://github.com/python/typing_extensions |
| urllib3 | 2.8.0 | MIT | https://github.com/urllib3/urllib3/blob/main/CHANGES.rst |
| utilsforecast | 0.2.15 | Apache Software License | https://github.com/Nixtla/utilsforecast |
| wrapt | 2.4.1 | BSD-2-Clause | https://github.com/GrahamDumpleton/wrapt |
| yarl | 1.25.1 | Apache-2.0 | https://github.com/aio-libs/yarl |
