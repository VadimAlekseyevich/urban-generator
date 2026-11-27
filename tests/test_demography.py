from dataclasses import FrozenInstanceError, asdict, replace
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from urban.generate import generate_buildings
from urban.io import read_terrain
from urban.model import DemographicProfile, Scenario
from urban.report import ReportValidationError, export_result_zip, validate_result
from urban.services import calculate_school_services


ROOT=Path(__file__).resolve().parents[1]


def completed(scenario=Scenario()):
    ground=read_terrain((ROOT/"examples/demo.tif").read_bytes())
    plan=generate_buildings(ground,scenario)
    return validate_result(plan,calculate_school_services(plan))


def test_default_and_legacy_scenario_keep_608_population_and_76_demand():
    legacy=asdict(Scenario());legacy.pop("demographic_profile")
    report=completed(Scenario(**legacy))
    assert report.metrics.population==608 and report.metrics.school_demand==76
    assert report.plan.scenario.demographic_profile==DemographicProfile()


def test_known_footprint_rounds_both_halves_up_and_exports_actual_profile():
    profile=DemographicProfile(30,.8125,.5,"Контрольное округление половин вверх")
    report=completed(Scenario(target_population=1,demographic_profile=profile))
    house=report.plan.buildings[0]
    assert house.footprint_area_m2==400 and house.floors==3
    assert house.population==33  # floor(400 * 3 * .8125 / 30 + .5) = floor(33)
    assert house.school_demand==17  # floor(33 * .5 + .5) = 17
    with ZipFile(BytesIO(export_result_zip(report))) as archive:
        metadata=json.loads(archive.read("metrics.json"))
        assert metadata["parameters"]["demographic_profile"]==asdict(profile)
        assert metadata["model_coefficients"]["useful_area_fraction"]==.8125
        assert metadata["model_coefficients"]["schoolchild_fraction"]==.5
        geo=json.loads(archive.read("project.geojson"))
        assert geo["urban_generator"]["parameters"]["demographic_profile"]==asdict(profile)


def test_schoolchild_fraction_changes_only_demand_and_school_services():
    baseline=completed()
    changed=completed(Scenario(demographic_profile=DemographicProfile(schoolchild_fraction=.25)))
    np.testing.assert_array_equal(baseline.plan.building_ids,changed.plan.building_ids)
    np.testing.assert_array_equal(baseline.plan.roads.road_mask,changed.plan.roads.road_mask)
    assert baseline.metrics.population==changed.metrics.population==608
    assert baseline.metrics.school_demand==76 and changed.metrics.school_demand==152


def test_zero_schoolchild_fraction_has_no_demand_or_percentage():
    report=completed(Scenario(demographic_profile=DemographicProfile(schoolchild_fraction=0)))
    assert report.metrics.population==608
    assert report.metrics.school_demand==report.metrics.opened_schools==0
    assert report.metrics.school_coverage_percent is None


@pytest.mark.parametrize("changes", [
    {"area_per_resident_m2":0},{"area_per_resident_m2":201},{"area_per_resident_m2":True},
    {"useful_area_fraction":0},{"useful_area_fraction":1.1},{"useful_area_fraction":float("nan")},
    {"schoolchild_fraction":-.01},{"schoolchild_fraction":1.1},{"schoolchild_fraction":float("inf")},
    {"source_note":""},{"source_note":" "},{"source_note":"x"*1001},{"source_note":None},
])
def test_invalid_profile_is_rejected_before_generation(changes):
    with pytest.raises(ValueError):
        DemographicProfile(**changes)


def test_profile_is_immutable_and_scenario_requires_validated_profile():
    with pytest.raises(FrozenInstanceError):
        DemographicProfile().schoolchild_fraction=.5
    with pytest.raises(ValueError,match="demographic_profile"):
        Scenario(demographic_profile={"schoolchild_fraction":.5})


def test_independent_validation_detects_stale_population_and_demand():
    report=completed()
    scenario=replace(report.plan.scenario,demographic_profile=DemographicProfile(useful_area_fraction=.9))
    with pytest.raises(ReportValidationError,match="населени|демограф"):
        validate_result(replace(report.plan,scenario=scenario),report.allocation)


def test_ui_error_keeps_completed_profile_and_variants_store_distinct_profiles():
    app=AppTest.from_file(str(ROOT/"app.py")).run(timeout=15)
    app.checkbox(key="legend_confirmed").check().run();app.button[0].click().run()
    app.button(key="pin_variant_a").click().run()
    original,data=app.session_state.verified_report,app.session_state.result_zip
    app.text_area(key="profile_note").set_value("")
    app.button[0].click().run()
    assert not app.exception and app.error
    assert app.session_state.verified_report is original and app.session_state.result_zip==data
    app.text_area(key="profile_note").set_value("Контрольная доля школьников")
    app.number_input(key="profile_schoolchildren").set_value(25)
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert app.session_state.variant_a.plan.scenario.demographic_profile.schoolchild_fraction==.12
    assert app.session_state.variant_b.plan.scenario.demographic_profile.schoolchild_fraction==.25
    assert app.session_state.verified_report.metrics.school_demand==152
    assert app.session_state.variant_a_zip==data
