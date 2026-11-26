# Приёмочная матрица MVP — §8 DESIGN.md

Проверка #18, v0.9. Воспроизведение из корня репозитория:
```sh
uv sync --frozen
uv run pytest -q
```

В таблице пути указаны относительно каталога tests/. Для точечного повтора
используйте `uv run pytest -q tests/имя.py::имя_теста`. Весь набор включает
параметризованные случаи; отдельное число тестов не равно числу строк матрицы.

| № | Контрольный вход | Проверки (файл::функция) | Ожидаемый результат |
|---|---|---|---|
| 1 | Демо GeoTIFF и отдельный обычный TIFF 10 м | test_io.py::test_demo_geotiff_and_equivalent_local_tiff; test_io.py::test_full_result_matches_between_geotiff_and_plain_tiff | Совпадают геометрия, ID, входы, назначения, матрица путей и метрики; только GeoTIFF имеет CRS, хеши разные |
| 2 | Класс 9, RGB, 257 столбцов, географическая CRS, NoData=0; ошибочный повтор | test_io.py::test_invalid_raster; test_io.py::test_geographic_crs_is_reported_even_with_degree_pixel_size; test_io.py::test_nodata_cannot_hide_a_semantic_class; test_app.py::test_invalid_replacement_cannot_relabel_completed_report; test_app.py::test_failed_regeneration_preserves_comparison_and_downloads | Ошибка до генерации; смена файла очищает отчёт, ошибочная генерация того же входа оставляет прошлый отчёт |
| 3 | Дорога и сплошная река | test_generate.py::test_river_keeps_new_roads_on_the_original_bank; test_buildings.py::test_demo_repeat_preserves_ids_entries_and_source_layers; test_report.py::test_demo_validates_all_metrics | Нет дорог и домов на изолированном берегу, исходные ячейки неизменны |
| 4 | Занятая территория или нет дороги | test_app.py::test_occupied_map_displays_zero_new_roads_without_error; test_report.py::test_empty_denominators; test_io.py::test_invalid_raster | Нулевой прирост и недобор 600; отсутствие дороги отклонено |
| 5 | Два дома 4+4, школа 5 мест | test_report.py::test_two_houses_coverage_and_deficit; test_school_allocation.py::test_zero_budget_keeps_all_demand_unserved; test_school_allocation.py::test_greedy_gain_uses_residual_demand_and_stops_without_gain | 5 обслужено, 3 дефицит, 62,5%; при бюджете 0 — 0; двойного учёта нет |
| 6 | U-образная дорога, порог 129,99/130 м | test_school_paths.py::test_u_path_and_inclusive_threshold; test_school_paths.py::test_u_threshold_controls_actual_assignments | Прямая дистанция меньше порога; сетевой путь 130 м, назначений 0/4 |
| 7 | Повтор, смена бюджета, этажности | test_report.py::test_full_report_repeat_preserves_classes_ids_and_metrics; test_buildings.py::test_budget_does_not_change_houses_but_floors_change_population_and_stopping; test_app.py::test_school_budget_comparison_through_form; test_app.py::test_floor_change_recalculates_population_and_preserves_roads | Полный повтор совпадает, включая ZIP; бюджет не меняет застройку; этажность меняет население и число домов |
| 8 | Экспорт двух координатных режимов | test_export.py::test_tiff_roundtrip_and_metadata; test_app.py::test_compare_budgets_and_download_completed_reports | Rasterio читает классы, маску, CRS/transform; CSV и JSON согласованы с интерфейсом |
| 9 | Испорченный результат | test_report.py::test_water_corruption_rejected; test_report.py::test_over_capacity_rejected; test_export.py::test_corrupt_raster_cannot_export; test_export.py::test_corrupt_assignment_cannot_export; test_export.py::test_corrupt_metrics_cannot_export | Ошибка валидатора блокирует выдачу ZIP |

## Фактический прогон

Результат CI фиксируется после выполнения полного набора.

## Ручные шаги, ещё не выполненные на ноутбуке защиты

Запустить `uv run streamlit run app.py`, открыть http://localhost:8501,
пройти сценарий из README, скачать ZIP и comparison.csv в браузере,
открыть result.tif в QGIS (или Rasterio). В AppTest уже проверены
байты, переданные кнопкам скачивания, но это не проверка самого браузера.
Реальный городской вход и производительность на ноутбуке защиты этой задачей
не измеряются. Парный эксперимент и замеры — задачи #19 и #20.
