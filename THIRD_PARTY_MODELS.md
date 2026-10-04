# Сторонние модели и библиотеки: лицензии и допустимость

Дата проверки: **2026-09-16**. Лицензии весов взяты из метаданных карточек Hugging Face
(`https://huggingface.co/api/models/<repo>`: поле `cardData.license` и тег `license:*`).
Лицензии библиотек взяты из метаданных установленных пакетов (`importlib.metadata`) и файлов LICENSE
в репозиториях.

**Правила оценки допустимости.** Положение конкурса запрещает платные проприетарные технологии без
open-source лицензии (п. 10.3). Победитель передаёт организатору простую неисключительную лицензию
на решение с правом сублицензирования, без ограничения по цели (п. 10.1). Поэтому допустимы только
компоненты под разрешительными лицензиями (Apache-2.0, MIT, BSD), в том числе для **весов**.
Non-commercial и кастомные лицензии весов, а также закрытые API не используются.

## Foundation-модели (веса)

| Модель | Код | Лицензия весов | Коммерческие ограничения | Допустима | Ревизия (закреплена в `configs/models.yaml`) | Источник |
|---|---|---|---|---|---|---|
| Chronos-2 (`amazon/chronos-2`) | Apache-2.0 (`chronos-forecasting` 2.3.2) | **Apache-2.0** | нет | ✅ да | `29ec3766d36d6f73f0696f85560a422f50e8498c` | https://huggingface.co/amazon/chronos-2 |
| TimesFM 2.5 200M (`google/timesfm-2.5-200m-pytorch`) | Apache-2.0 (`timesfm` 3.0.2, класс `TimesFM_2p5_200M_torch`) | **Apache-2.0** | нет | ✅ да | `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` | https://huggingface.co/google/timesfm-2.5-200m-pytorch |

## Исключённые компоненты

| Компонент | Код | Лицензия весов / доступ | Причина исключения | Источник |
|---|---|---|---|---|
| TimesFM 3.0 (`google/timesfm-3.0-pytorch`) | Apache-2.0 | `timesfm-non-commercial-license-v1.0` | non-commercial веса несовместимы с п. 10.1 (лицензия организатору без ограничения цели) | https://huggingface.co/google/timesfm-3.0-pytorch |
| Moirai-2 (`Salesforce/moirai-2.0-R-small`), Moirai / Moirai-MoE | Apache-2.0 (uni2ts) | `cc-by-nc-4.0` | non-commercial веса | https://huggingface.co/Salesforce/moirai-2.0-R-small |
| TiRex (`NX-AI/TiRex`) | NXAI Community License | `nx-ai-community-license` | кастомная лицензия с ограничениями на код и веса | https://huggingface.co/NX-AI/TiRex |
| TabPFN-TS (веса `Prior-Labs/TabPFN-v2-*`) | Apache-2.0 (`tabpfn-time-series`) | `priorlabs-1-1` | кастомная лицензия весов Prior Labs с ограничениями | https://huggingface.co/Prior-Labs/TabPFN-v2-reg |
| TimeGPT (Nixtla) | SDK Apache-2.0 | закрытый платный API, веса не публикуются | проприетарная платная технология (п. 10.3) | https://www.nixtla.io/docs |

Пакет `timesfm` 3.0.2 содержит классы и для 2.5, и для 3.0. Используется **только**
`TimesFM_2p5_200M_torch` с весами `google/timesfm-2.5-200m-pytorch`. Загрузка весов 3.0 в коде
и конфигурации запрещена и проверяется тестом `tests/shell/test_models_config.py`.

## Библиотеки

| Библиотека | Версия | Лицензия | Допустима |
|---|---|---|---|
| statsforecast | 2.0.1 | Apache-2.0 | ✅ |
| prophet | 1.4.0 | MIT | ✅ |
| lightgbm | 4.7.0 | MIT | ✅ |
| neuralforecast | 3.1.7 | Apache-2.0 | ✅ |
| chronos-forecasting | 2.3.2 | Apache-2.0 | ✅ |
| timesfm | 3.0.2 | Apache-2.0 (код) | ✅ (только веса 2.5) |
| ruptures | 1.1.10 | BSD-2-Clause | ✅ |
| river | 0.26.1 | BSD-3-Clause | ✅ |
| optuna | 5.0.0 | MIT | ✅ |
| torch | 2.14.0 | BSD-3-Clause (с компонентами Apache-2.0) | ✅ |
| transformers | 5.17.0 | Apache-2.0 | ✅ |
| pandas | 3.0.5 | BSD-3-Clause | ✅ |
| numpy | 2.5.3 | BSD-3-Clause | ✅ |
| scipy | 1.18.1 | BSD-3-Clause | ✅ |
| scikit-learn | 1.9.1 | BSD-3-Clause | ✅ |
| pyarrow | 25.0.1 | Apache-2.0 | ✅ |
| utilsforecast | 0.2.15 | Apache-2.0 | ✅ |
| matplotlib | 3.11.2 | PSF-based (matplotlib license) | ✅ |
| pyyaml | 6.0.3 | MIT | ✅ |
| typer | 0.27.2 | MIT | ✅ |

Полный автоматический список лицензий всех транзитивных зависимостей формируется в
`THIRD_PARTY_LICENSES.md` перед подачей.

## Данные

| Источник | Лицензия / условия | Использование |
|---|---|---|
| СберИндекс (sberindex.ru) | открытые данные, при копировании необходимо упоминание источника | основной источник; указание «Данные СберИндекса» в README, отчёте и лендинге |
| GDELT Events | открытые данные GDELT (свободное использование с указанием источника) | новостные признаки; в репозиторий попадают только агрегаты и URL |
