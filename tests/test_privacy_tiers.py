"""
Test script for AURA v3 Privacy Tier System
"""

from src.core.privacy_tiers import (
    PrivacyTier,
    UserPermissionManager,
    get_permission_manager,
    CategoryRegistry,
)


def test_category_registry():
    """Test category registry"""
    print("Testing CategoryRegistry...")
    registry = CategoryRegistry()

    # Test getting category
    cat = registry.get_category("gallery_photos")
    assert cat is not None
    assert cat.default_tier == PrivacyTier.PRIVATE
    print(f"  ✓ gallery_photos default tier: {cat.default_tier.value}")

    # Test keyword search
    found = registry.find_category_by_keyword("photo")
    assert found == "gallery_photos"
    print(f"  ✓ Keyword 'photo' -> {found}")

    # Test default for unknown
    tier = registry.get_tier("unknown_category")
    assert tier == PrivacyTier.PUBLIC
    print(f"  ✓ Unknown category defaults to: {tier.value}")


def test_permission_manager():
    """Test permission manager"""
    print("\nTesting UserPermissionManager...")
    pm = UserPermissionManager()

    # Test default tier
    assert pm._default_tier == PrivacyTier.SENSITIVE
    print(f"  ✓ Default tier: {pm._default_tier.value}")

    # Test tier override
    pm.set_tier_override("gallery_photos", PrivacyTier.PUBLIC)
    tier = pm.get_effective_tier("gallery_photos")
    assert tier == PrivacyTier.PUBLIC
    print(f"  ✓ Override works: gallery_photos -> {tier.value}")

    # Clear override
    pm.clear_override("gallery_photos")
    tier = pm.get_effective_tier("gallery_photos")
    assert tier == PrivacyTier.PRIVATE
    print(f"  ✓ Clear override: gallery_photos -> {tier.value}")


def test_permission_checking():
    """Test permission checking logic"""
    print("\nTesting permission checking...")
    pm = get_permission_manager()

    # PRIVATE tier always requires confirmation
    req = pm.check_permission("gallery_photos", "read")
    assert req.context.get("needs_confirmation") == True
    print(f"  ✓ PRIVATE read requires confirmation")

    req = pm.check_permission("gallery_photos", "write")
    assert req.context.get("needs_confirmation") == True
    print(f"  ✓ PRIVATE write requires confirmation")

    # SENSITIVE tier: confirm for writes
    req = pm.check_permission("banking_apps", "write")
    assert req.context.get("needs_confirmation") == True
    print(f"  ✓ SENSITIVE write requires confirmation")

    # PUBLIC tier: no confirmation needed
    req = pm.check_permission("news_apps", "read")
    assert req.context.get("needs_confirmation") == False
    print(f"  ✓ PUBLIC read doesn't require confirmation")


def test_natural_language():
    """Test natural language permission parsing"""
    print("\nTesting natural language parsing...")
    pm = get_permission_manager()

    tests = [
        ("You can access my photos", ("gallery_photos", "session")),
        ("Don't read my messages", ("messages", "restrict")),
        ("Feel free to check my calendar", ("calendar", "session")),
    ]

    for text, expected in tests:
        result = pm.parse_natural_permission(text)
        assert result == expected, f"Failed: {text} -> {result} (expected {expected})"
        print(f"  ✓ '{text[:30]}...' -> {result}")


def test_grant_permission():
    """Test permission granting"""
    print("\nTesting permission grants...")
    pm = get_permission_manager()

    # Grant session permission
    pm.grant_permission("gallery_photos", scope="session")
    grant = pm._grants.get("gallery_photos")
    assert grant is not None
    assert grant.scope == "session"
    print(f"  ✓ Session grant created")

    # Revoke
    pm.revoke_permission("gallery_photos")
    assert "gallery_photos" not in pm._grants
    print(f"  ✓ Permission revoked")


def test_status():
    """Test status reporting"""
    print("\nTesting status reporting...")
    pm = get_permission_manager()

    status = pm.get_permission_status()
    assert "default_tier" in status
    assert "categories" in status
    assert len(status["categories"]) > 0
    print(f"  ✓ Status has {len(status['categories'])} categories")


if __name__ == "__main__":
    print("=" * 50)
    print("AURA v3 Privacy Tier System Tests")
    print("=" * 50)

    test_category_registry()
    test_permission_manager()
    test_permission_checking()
    test_natural_language()
    test_grant_permission()
    test_status()

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
