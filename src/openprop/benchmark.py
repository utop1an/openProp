from __future__ import annotations

from dataclasses import dataclass

from .models import (
    Entity,
    Observation,
    PropertyConstraint,
    PropertyDefinition,
    QueryFrame,
    RelationValue,
    ValueType,
)
from .property_registry import PropertyRegistry


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    query: str
    entities: tuple[Entity, ...]
    target_id: str
    gold_frame: QueryFrame
    tags: tuple[str, ...] = ()


def core_registry() -> PropertyRegistry:
    registry = PropertyRegistry()
    definitions = (
        PropertyDefinition("type", "semantic object category", ValueType.SEMANTIC),
        PropertyDefinition("color", "perceived surface color", ValueType.SEMANTIC),
        PropertyDefinition(
            "location",
            "spatial relation between an entity and another entity",
            ValueType.RELATION,
            aliases=("spatial relation", "position relation"),
            metadata={"argument_roles": ["object"]},
        ),
        PropertyDefinition("material", "primary physical material", ValueType.SEMANTIC),
        PropertyDefinition("owner", "person associated with the entity", ValueType.ENTITY_REFERENCE),
        PropertyDefinition(
            "temperature",
            "surface temperature in degrees Celsius",
            ValueType.NUMERIC,
            unit="celsius",
            metadata={"scale": 10.0},
        ),
        PropertyDefinition(
            "size",
            "largest physical dimension in centimetres",
            ValueType.NUMERIC,
            unit="centimetres",
            metadata={"scale": 5.0},
        ),
        PropertyDefinition(
            "weight",
            "physical weight in kilograms",
            ValueType.NUMERIC,
            unit="kilograms",
            metadata={"scale": 2.0},
        ),
    )
    for definition in definitions:
        registry.register(definition)
    return registry


def _relation(predicate: str, object_id: str) -> RelationValue:
    return RelationValue(predicate, {"object": object_id})


def _entity(entity_id: str, **properties: object) -> Entity:
    observations = {
        name: value if isinstance(value, Observation) else Observation(value)
        for name, value in properties.items()
    }
    return Entity(entity_id, observations)


def _constraint(
    name: str,
    value: object,
    relevance: float,
    tolerance: float | None = None,
) -> PropertyConstraint:
    return PropertyConstraint(name, value, relevance, tolerance)


def _case(
    case_id: str,
    query: str,
    entities: tuple[Entity, ...],
    target_id: str,
    *constraints: PropertyConstraint,
    tags: tuple[str, ...] = (),
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id,
        query,
        entities,
        target_id,
        QueryFrame(query, tuple(constraints)),
        tags,
    )


def core_benchmark() -> tuple[BenchmarkCase, ...]:
    """Thirty bilingual, typed entity-reference cases across six scenes."""

    desk = (
        _entity("desk_red_cup", type="cup", color="red", material="ceramic", location=_relation("on", "table"), size=10),
        _entity("desk_blue_cup", type="cup", color="blue", material="ceramic", location=_relation("on", "table"), size=10),
        _entity("shelf_red_bowl", type="bowl", color="red", material="ceramic", location=_relation("on", "shelf"), size=16),
        _entity("cabinet_red_cup", type="cup", color="red", material="plastic", location=_relation("inside", "cabinet"), size=12),
    )
    kitchen = (
        _entity("steel_kettle", type="kettle", color="silver", material="steel", location=_relation("on", "stove"), temperature=95, size=25),
        _entity("warm_mug", type="mug", color="red", material="ceramic", location=_relation("on", "counter"), temperature=42, size=10),
        _entity("cold_bottle", type="bottle", color="blue", material="plastic", location=_relation("inside", "refrigerator"), temperature=5, size=22),
        _entity("black_pan", type="pan", color="black", material="iron", location=_relation("on", "stove"), temperature=80, size=30),
    )
    wardrobe = (
        _entity("alice_coat", type="coat", color="red", material="wool", owner="alice", location=_relation("on", "hook")),
        _entity("bob_shirt", type="shirt", color="blue", material="cotton", owner="bob", location=_relation("inside", "closet")),
        _entity("alice_shoe", type="shoe", color="black", material="leather", owner="alice", location=_relation("on", "floor")),
        _entity("carol_scarf", type="scarf", color="red", material="silk", owner="carol", location=_relation("inside", "drawer")),
    )
    office = (
        _entity("alice_report", type="document", color="white", material="paper", owner="alice", location=_relation("on", "desk")),
        _entity("bob_notebook", type="notebook", color="blue", material="paper", owner="bob", location=_relation("on", "shelf")),
        _entity("alice_key", type="key", color="silver", material="steel", owner="alice", location=_relation("inside", "drawer")),
        _entity("carol_phone", type="phone", color="black", material="glass", owner="carol", location=_relation("on", "desk")),
    )
    warehouse = (
        _entity("small_red_box", type="box", color="red", material="cardboard", location=_relation("on", "rack_a"), size=20, weight=2),
        _entity("large_red_box", type="box", color="red", material="cardboard", location=_relation("on", "rack_b"), size=80, weight=12),
        _entity("blue_crate", type="crate", color="blue", material="plastic", location=_relation("on", "rack_a"), size=60, weight=8),
        _entity("steel_case", type="case", color="silver", material="steel", location=_relation("on", "floor"), size=50, weight=20),
    )
    garden = (
        _entity("red_rose", type="flower", color="red", location=_relation("inside", "pot_a"), size=35, owner="alice"),
        _entity("yellow_tulip", type="flower", color="yellow", location=_relation("inside", "pot_b"), size=30, owner="bob"),
        _entity("small_shovel", type="shovel", color="green", material="steel", location=_relation("beside", "shed"), size=45, owner="alice"),
        _entity("watering_can", type="watering can", color="blue", material="plastic", location=_relation("inside", "shed"), size=40, owner="carol"),
    )

    return (
        _case("desk-01", "桌上的红色杯子", desk, "desk_red_cup", _constraint("type", "cup", .95), _constraint("color", "red", .93), _constraint("location", _relation("on", "table"), .97), tags=("zh", "relation")),
        _case("desk-02", "the blue cup on the table", desk, "desk_blue_cup", _constraint("type", "cup", .95), _constraint("color", "blue", .95), _constraint("location", _relation("on", "table"), .9), tags=("en", "relation")),
        _case("desk-03", "架子上的红碗", desk, "shelf_red_bowl", _constraint("type", "bowl", .95), _constraint("color", "red", .85), _constraint("location", _relation("on", "shelf"), .95), tags=("zh", "relation")),
        _case("desk-04", "柜子里的塑料杯", desk, "cabinet_red_cup", _constraint("type", "cup", .9), _constraint("material", "plastic", .95), _constraint("location", _relation("inside", "cabinet"), .95), tags=("zh", "relation")),
        _case("desk-05", "the larger red ceramic object", desk, "shelf_red_bowl", _constraint("color", "red", .75), _constraint("material", "ceramic", .8), _constraint("size", 16, .95, 3), tags=("en", "numeric")),
        _case("kitchen-01", "炉子上的银色水壶", kitchen, "steel_kettle", _constraint("type", "kettle", .95), _constraint("color", "silver", .8), _constraint("location", _relation("on", "stove"), .9), tags=("zh", "relation")),
        _case("kitchen-02", "the warm red mug on the counter", kitchen, "warm_mug", _constraint("type", "mug", .95), _constraint("color", "red", .8), _constraint("temperature", 42, .85, 10), _constraint("location", _relation("on", "counter"), .8), tags=("en", "numeric")),
        _case("kitchen-03", "冰箱里的蓝色塑料瓶", kitchen, "cold_bottle", _constraint("type", "bottle", .95), _constraint("color", "blue", .8), _constraint("material", "plastic", .75), _constraint("location", _relation("inside", "refrigerator"), .9), tags=("zh", "relation")),
        _case("kitchen-04", "the hot black iron pan", kitchen, "black_pan", _constraint("type", "pan", .95), _constraint("color", "black", .8), _constraint("material", "iron", .8), _constraint("temperature", 80, .75, 15), tags=("en", "numeric")),
        _case("kitchen-05", "温度接近九十五度的钢制物体", kitchen, "steel_kettle", _constraint("temperature", 95, .98, 8), _constraint("material", "steel", .9), tags=("zh", "numeric")),
        _case("wardrobe-01", "爱丽丝挂在钩子上的红外套", wardrobe, "alice_coat", _constraint("owner", "alice", .9), _constraint("type", "coat", .95), _constraint("color", "red", .8), _constraint("location", _relation("on", "hook"), .85), tags=("zh", "owner")),
        _case("wardrobe-02", "Bob's blue cotton shirt", wardrobe, "bob_shirt", _constraint("owner", "bob", .95), _constraint("type", "shirt", .95), _constraint("color", "blue", .75), _constraint("material", "cotton", .8), tags=("en", "owner")),
        _case("wardrobe-03", "地板上爱丽丝的黑色皮鞋", wardrobe, "alice_shoe", _constraint("owner", "alice", .85), _constraint("type", "shoe", .95), _constraint("color", "black", .8), _constraint("material", "leather", .8), _constraint("location", _relation("on", "floor"), .85), tags=("zh", "owner")),
        _case("wardrobe-04", "Carol's red silk item in the drawer", wardrobe, "carol_scarf", _constraint("owner", "carol", .9), _constraint("color", "red", .8), _constraint("material", "silk", .9), _constraint("location", _relation("inside", "drawer"), .9), tags=("en", "owner")),
        _case("wardrobe-05", "抽屉里的围巾", wardrobe, "carol_scarf", _constraint("type", "scarf", .95), _constraint("location", _relation("inside", "drawer"), .9), tags=("zh", "relation")),
        _case("office-01", "桌上爱丽丝的白色文件", office, "alice_report", _constraint("owner", "alice", .9), _constraint("type", "document", .9), _constraint("color", "white", .75), _constraint("location", _relation("on", "desk"), .85), tags=("zh", "owner")),
        _case("office-02", "Bob's blue notebook on the shelf", office, "bob_notebook", _constraint("owner", "bob", .9), _constraint("type", "notebook", .95), _constraint("color", "blue", .8), _constraint("location", _relation("on", "shelf"), .9), tags=("en", "owner")),
        _case("office-03", "爱丽丝放在抽屉里的钢钥匙", office, "alice_key", _constraint("owner", "alice", .85), _constraint("type", "key", .95), _constraint("material", "steel", .8), _constraint("location", _relation("inside", "drawer"), .9), tags=("zh", "owner")),
        _case("office-04", "the black glass phone on the desk", office, "carol_phone", _constraint("type", "phone", .95), _constraint("color", "black", .8), _constraint("material", "glass", .8), _constraint("location", _relation("on", "desk"), .85), tags=("en", "relation")),
        _case("office-05", "Carol's item on the desk", office, "carol_phone", _constraint("owner", "carol", .95), _constraint("location", _relation("on", "desk"), .8), tags=("en", "owner")),
        _case("warehouse-01", "A架上的小红纸箱", warehouse, "small_red_box", _constraint("type", "box", .85), _constraint("color", "red", .75), _constraint("material", "cardboard", .75), _constraint("location", _relation("on", "rack_a"), .9), _constraint("size", 20, .9, 8), tags=("zh", "numeric")),
        _case("warehouse-02", "the large red box on rack B", warehouse, "large_red_box", _constraint("type", "box", .9), _constraint("color", "red", .75), _constraint("location", _relation("on", "rack_b"), .9), _constraint("size", 80, .95, 10), tags=("en", "numeric")),
        _case("warehouse-03", "A架上的蓝色塑料箱", warehouse, "blue_crate", _constraint("type", "crate", .85), _constraint("color", "blue", .8), _constraint("material", "plastic", .8), _constraint("location", _relation("on", "rack_a"), .9), tags=("zh", "relation")),
        _case("warehouse-04", "the heavy steel case on the floor", warehouse, "steel_case", _constraint("type", "case", .9), _constraint("material", "steel", .9), _constraint("location", _relation("on", "floor"), .85), _constraint("weight", 20, .9, 4), tags=("en", "numeric")),
        _case("warehouse-05", "重量约十二公斤的红纸箱", warehouse, "large_red_box", _constraint("weight", 12, .98, 3), _constraint("color", "red", .7), _constraint("material", "cardboard", .8), tags=("zh", "numeric")),
        _case("garden-01", "A花盆里的红花", garden, "red_rose", _constraint("type", "flower", .9), _constraint("color", "red", .85), _constraint("location", _relation("inside", "pot_a"), .95), tags=("zh", "relation")),
        _case("garden-02", "Bob's yellow flower in pot B", garden, "yellow_tulip", _constraint("owner", "bob", .8), _constraint("type", "flower", .9), _constraint("color", "yellow", .85), _constraint("location", _relation("inside", "pot_b"), .95), tags=("en", "owner")),
        _case("garden-03", "棚子旁边爱丽丝的小铲子", garden, "small_shovel", _constraint("owner", "alice", .8), _constraint("type", "shovel", .95), _constraint("location", _relation("beside", "shed"), .9), _constraint("size", 45, .65, 8), tags=("zh", "relation")),
        _case("garden-04", "the blue plastic watering can inside the shed", garden, "watering_can", _constraint("type", "watering can", .95), _constraint("color", "blue", .8), _constraint("material", "plastic", .8), _constraint("location", _relation("inside", "shed"), .9), tags=("en", "relation")),
        _case("garden-05", "Carol's object in the shed", garden, "watering_can", _constraint("owner", "carol", .95), _constraint("location", _relation("inside", "shed"), .85), tags=("en", "owner")),
    )
