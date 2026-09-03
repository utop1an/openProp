from __future__ import annotations

from dataclasses import replace

from .models import QueryFrame
from .temporal_grounding import TemporalGroundingCase, temporal_grounding_benchmark


RELATION_QUERIES = (
    "请找出放在桌面上的红杯子",
    "find the red cup resting on the table",
    "我要桌子上那个红色的杯子",
    "which red cup is currently on the tabletop",
    "桌面上哪一个杯子是红色的",
    "select the cup that is red and on the table",
    "指向仍在桌上的红色杯子",
    "point to the red cup positioned on the table",
    "定位桌上那只红杯",
    "locate the red-colored cup atop the table",
)

CLEAN_QUERIES = (
    "找出现在仍然干净的蓝色衬衫",
    "find the blue shirt that is still clean",
    "哪一件蓝衬衫目前是干净的",
    "select the currently clean blue shirt",
    "请定位那件干净的蓝色衬衣",
    "point to the blue-colored shirt that remains clean",
    "我要状态为干净的那件蓝衬衫",
    "which shirt is both blue and clean now",
    "指出当前干净的蓝色上衣",
    "locate the clean shirt with the blue color",
)

CONTROL_QUERIES = (
    "请找出红色陶瓷材质的杯子",
    "find the ceramic cup that is red",
    "我要那只红色的陶瓷杯",
    "select the red cup made of ceramic",
    "哪一个杯子既是红色又是陶瓷的",
    "point to the cup with red color and ceramic material",
    "定位红色陶瓷杯具",
    "which ceramic drinking cup is red",
    "请指出材质为陶瓷的红杯子",
    "locate the red-colored ceramic cup",
)


def paraphrased_temporal_grounding_benchmark() -> tuple[TemporalGroundingCase, ...]:
    """Return 40 cases using 30 queries disjoint from development templates."""

    cases = temporal_grounding_benchmark(repetitions=10)
    rewritten: list[TemporalGroundingCase] = []
    for case in cases:
        index = int(case.case_id.rsplit("-", 1)[1])
        if "event-invalidated" in case.tags:
            query = CLEAN_QUERIES[index]
        elif "static-control" in case.tags:
            query = CONTROL_QUERIES[index]
        else:
            query = RELATION_QUERIES[index]
        rewritten.append(
            replace(
                case,
                query=query,
                gold_frame=QueryFrame(query, case.gold_frame.constraints),
            )
        )
    return tuple(rewritten)
