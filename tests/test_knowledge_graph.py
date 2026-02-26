"""
Tests for AURA v3 Knowledge Graph

Tests the GraphPilot-style knowledge graph implementation
for offline app topology mapping.
"""

import unittest
import tempfile
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory.knowledge_graph import (
    KnowledgeGraph,
    AppNode,
    RelationshipEdge,
    RelationshipType,
    ValidityInterval,
    TopologyMapper,
    QueryEngine,
    get_knowledge_graph,
)


class TestValidityInterval(unittest.TestCase):
    """Test ValidityInterval class"""

    def test_is_valid_current_time(self):
        interval = ValidityInterval(start=1000.0)
        self.assertTrue(interval.is_valid())

    def test_is_valid_past_time(self):
        interval = ValidityInterval(start=1000.0, end=2000.0)
        self.assertTrue(interval.is_valid(1500.0))

    def test_is_valid_expired(self):
        interval = ValidityInterval(start=1000.0, end=2000.0)
        self.assertFalse(interval.is_valid(2500.0))

    def test_is_valid_future(self):
        interval = ValidityInterval(start=3000.0)
        self.assertFalse(interval.is_valid(1000.0))

    def test_to_from_dict(self):
        original = ValidityInterval(start=1000.0, end=2000.0)
        data = original.to_dict()
        restored = ValidityInterval.from_dict(data)
        self.assertEqual(restored.start, original.start)
        self.assertEqual(restored.end, original.end)


class TestAppNode(unittest.TestCase):
    """Test AppNode class"""

    def test_create_node(self):
        node = AppNode(
            id="com.example.app",
            name="Example App",
            package_name="com.example.app",
            capabilities=["camera", "storage"],
            permissions=["android.permission.CAMERA"],
            category="productivity",
        )
        self.assertEqual(node.id, "com.example.app")
        self.assertEqual(node.name, "Example App")
        self.assertEqual(len(node.capabilities), 2)
        self.assertEqual(node.use_count, 0)

    def test_to_from_dict(self):
        original = AppNode(
            id="com.test.app",
            name="Test App",
            package_name="com.test.app",
            capabilities=["network", "storage"],
            category="utility",
            is_running=True,
            use_count=5,
        )
        data = original.to_dict()
        restored = AppNode.from_dict(data)

        self.assertEqual(restored.id, original.id)
        self.assertEqual(restored.name, original.name)
        self.assertEqual(restored.capabilities, original.capabilities)
        self.assertEqual(restored.is_running, original.is_running)
        self.assertEqual(restored.use_count, original.use_count)


class TestRelationshipEdge(unittest.TestCase):
    """Test RelationshipEdge class"""

    def test_create_edge(self):
        edge = RelationshipEdge(
            id="rel_1",
            source_id="app1",
            target_id="app2",
            relationship_type=RelationshipType.COMMUNICATES_WITH,
            weight=0.8,
            frequency=10,
        )
        self.assertEqual(edge.source_id, "app1")
        self.assertEqual(edge.target_id, "app2")
        self.assertEqual(edge.relationship_type, RelationshipType.COMMUNICATES_WITH)
        self.assertEqual(edge.weight, 0.8)

    def test_to_from_dict(self):
        original = RelationshipEdge(
            id="rel_test",
            source_id="source_app",
            target_id="target_app",
            relationship_type=RelationshipType.DATA_SHARE,
            weight=0.5,
            frequency=3,
        )
        data = original.to_dict()
        restored = RelationshipEdge.from_dict(data)

        self.assertEqual(restored.id, original.id)
        self.assertEqual(restored.source_id, original.source_id)
        self.assertEqual(restored.relationship_type, original.relationship_type)


class TestKnowledgeGraph(unittest.TestCase):
    """Test KnowledgeGraph class"""

    def setUp(self):
        """Create temporary database for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_kg.db")
        self.kg = KnowledgeGraph(self.db_path)

    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        from src.utils.db_pool import close_all

        close_all()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_add_node(self):
        node = AppNode(
            id="com.whatsapp",
            name="WhatsApp",
            package_name="com.whatsapp",
            capabilities=["camera", "microphone", "storage"],
            permissions=["android.permission.CAMERA"],
            category="communication",
        )
        result = self.kg.add_node(node)

        self.assertEqual(result.id, "com.whatsapp")
        self.assertEqual(len(self.kg._node_cache), 1)

    def test_get_node(self):
        node = AppNode(
            id="com.test.get",
            name="Test Get",
            package_name="com.test.get",
        )
        self.kg.add_node(node)

        retrieved = self.kg.get_node("com.test.get")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Test Get")

    def test_add_relationship(self):
        # Add two nodes
        node1 = AppNode(id="app1", name="App 1", package_name="com.app1")
        node2 = AppNode(id="app2", name="App 2", package_name="com.app2")
        self.kg.add_node(node1)
        self.kg.add_node(node2)

        # Add relationship
        edge = RelationshipEdge(
            id="rel_1",
            source_id="app1",
            target_id="app2",
            relationship_type=RelationshipType.COMMUNICATES_WITH,
        )
        self.kg.add_relationship(edge)

        # Verify relationship
        rels = self.kg.get_relationships("app1")
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0].target_id, "app2")

    def test_query_by_capability(self):
        # Add nodes with capabilities
        apps = [
            AppNode(
                id="app_camera",
                name="Camera App",
                package_name="com.camera",
                capabilities=["camera", "storage"],
            ),
            AppNode(
                id="app_gallery",
                name="Gallery App",
                package_name="com.gallery",
                capabilities=["storage", "media"],
            ),
            AppNode(
                id="app_browser",
                name="Browser App",
                package_name="com.browser",
                capabilities=["network"],
            ),
        ]

        for app in apps:
            self.kg.add_node(app)

        # Query by capability
        camera_apps = self.kg.query_by_capability("camera")
        self.assertEqual(len(camera_apps), 1)
        self.assertEqual(camera_apps[0].id, "app_camera")

        storage_apps = self.kg.query_by_capability("storage")
        self.assertEqual(len(storage_apps), 2)

    def test_query_by_data_type(self):
        # Add nodes with data types
        apps = [
            AppNode(
                id="app_contacts",
                name="Contacts",
                package_name="com.contacts",
                data_types=["contacts"],
                export_data_schemas=["contacts"],
            ),
            AppNode(
                id="app_messages",
                name="Messages",
                package_name="com.messages",
                data_types=["contacts", "messages"],
                import_data_schemas=["contacts"],
            ),
        ]

        for app in apps:
            self.kg.add_node(app)

        # Query by data type
        contacts_apps = self.kg.query_by_data_type("contacts")
        self.assertEqual(len(contacts_apps), 2)

    def test_query_path(self):
        # Create a path: A -> B -> C
        apps = [
            AppNode(id="app_a", name="App A", package_name="com.a"),
            AppNode(id="app_b", name="App B", package_name="com.b"),
            AppNode(id="app_c", name="App C", package_name="com.c"),
        ]

        for app in apps:
            self.kg.add_node(app)

        # Add relationships
        self.kg.add_relationship(
            RelationshipEdge(
                id="rel_ab",
                source_id="app_a",
                target_id="app_b",
                relationship_type=RelationshipType.COMMUNICATES_WITH,
            )
        )
        self.kg.add_relationship(
            RelationshipEdge(
                id="rel_bc",
                source_id="app_b",
                target_id="app_c",
                relationship_type=RelationshipType.COMMUNICATES_WITH,
            )
        )

        # Query path
        path = self.kg.query_path("app_a", "app_c")
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 3)
        self.assertEqual(path, ["app_a", "app_b", "app_c"])

    def test_get_stats(self):
        node = AppNode(
            id="stats_app",
            name="Stats App",
            package_name="com.stats",
            capabilities=["test"],
        )
        self.kg.add_node(node)

        stats = self.kg.get_stats()
        self.assertEqual(stats["total_nodes"], 1)
        self.assertEqual(stats["active_nodes"], 1)


class TestTopologyMapper(unittest.TestCase):
    """Test TopologyMapper class"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_mapper.db")
        self.kg = KnowledgeGraph(self.db_path)
        self.mapper = TopologyMapper(self.kg)

    def tearDown(self):
        import shutil
        from src.utils.db_pool import close_all

        close_all()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_infer_capability_relationships(self):
        # Add nodes with capabilities
        apps = [
            AppNode(
                id="app1",
                name="App 1",
                package_name="com.app1",
                capabilities=["camera"],
            ),
            AppNode(
                id="app2",
                name="App 2",
                package_name="com.app2",
                capabilities=["camera", "storage"],
            ),
        ]

        for app in apps:
            self.kg.add_node(app)

        # Infer relationships
        self.mapper._infer_capability_relationships()

        # Check relationships were created
        rels = self.kg.get_relationships("app1", RelationshipType.PROVIDES_CAPABILITY)
        self.assertTrue(len(rels) > 0)


class TestQueryEngine(unittest.TestCase):
    """Test QueryEngine class"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_query.db")
        self.kg = KnowledgeGraph(self.db_path)
        self.query_engine = QueryEngine(self.kg)

    def tearDown(self):
        import shutil
        from src.utils.db_pool import close_all

        close_all()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_find_capable_apps(self):
        # Add test apps
        apps = [
            AppNode(
                id="whatsapp",
                name="WhatsApp",
                package_name="com.whatsapp",
                capabilities=["camera", "microphone"],
                use_count=50,
                is_running=True,
            ),
            AppNode(
                id="telegram",
                name="Telegram",
                package_name="org.telegram",
                capabilities=["camera", "microphone"],
                use_count=20,
            ),
            AppNode(
                id="signal",
                name="Signal",
                package_name="org.signal",
                capabilities=["camera", "microphone"],
                use_count=10,
            ),
        ]

        for app in apps:
            self.kg.add_node(app)

        # Query capable apps
        camera_apps = self.query_engine.find_capable_apps("camera")

        # Should return sorted by use_count
        self.assertEqual(len(camera_apps), 3)
        self.assertEqual(camera_apps[0].id, "whatsapp")  # Most used

    def test_recommend_tool_binding(self):
        # Add test apps
        apps = [
            AppNode(
                id="camera_app",
                name="Camera",
                package_name="com.camera",
                capabilities=["camera"],
                use_count=100,
                is_running=True,
            ),
            AppNode(
                id="gallery_app",
                name="Gallery",
                package_name="com.gallery",
                capabilities=["camera"],  # Can also access camera
                use_count=10,
            ),
        ]

        for app in apps:
            self.kg.add_node(app)

        # Get recommendation
        recommendation = self.query_engine.recommend_tool_binding("camera")

        self.assertIsNotNone(recommendation["recommended"])
        self.assertEqual(recommendation["recommended"], "camera_app")
        self.assertGreater(recommendation["confidence"], 0.5)

    def test_get_app_dependencies(self):
        # Add apps with relationships
        apps = [
            AppNode(id="app_main", name="Main App", package_name="com.main"),
            AppNode(id="app_dep", name="Dependency", package_name="com.dep"),
            AppNode(id="app_comm", name="Communicator", package_name="com.comm"),
        ]

        for app in apps:
            self.kg.add_node(app)

        # Add relationships
        self.kg.add_relationship(
            RelationshipEdge(
                id="dep_rel",
                source_id="app_main",
                target_id="app_dep",
                relationship_type=RelationshipType.DEPENDS_ON,
            )
        )
        self.kg.add_relationship(
            RelationshipEdge(
                id="comm_rel",
                source_id="app_main",
                target_id="app_comm",
                relationship_type=RelationshipType.COMMUNICATES_WITH,
            )
        )

        # Get dependencies
        deps = self.query_engine.get_app_dependencies("app_main")

        self.assertIn("app_dep", deps["depends_on"])
        self.assertIn("app_comm", deps["communicates_with"])


class TestIntegration(unittest.TestCase):
    """Integration tests for knowledge graph"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_integration.db")

    def tearDown(self):
        import shutil
        from src.utils.db_pool import close_all

        close_all()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_full_workflow(self):
        """Test complete workflow: create nodes, relationships, queries"""
        kg = KnowledgeGraph(self.db_path)

        # Create apps
        whatsapp = AppNode(
            id="com.whatsapp",
            name="WhatsApp",
            package_name="com.whatsapp",
            capabilities=["camera", "microphone", "storage"],
            permissions=["android.permission.CAMERA"],
            category="communication",
            use_count=100,
            is_running=True,
        )

        gallery = AppNode(
            id="com.gallery",
            name="Gallery",
            package_name="com.gallery",
            capabilities=["storage", "media"],
            category="media",
            use_count=50,
        )

        # Add to graph
        kg.add_node(whatsapp)
        kg.add_node(gallery)

        # Add relationship
        kg.add_relationship(
            RelationshipEdge(
                id="share_photos",
                source_id="com.whatsapp",
                target_id="com.gallery",
                relationship_type=RelationshipType.DATA_SHARE,
                weight=0.8,
            )
        )

        # Query
        capable = kg.query_by_capability("camera")
        self.assertEqual(len(capable), 1)

        # Query engine
        qe = QueryEngine(kg)
        recommendation = qe.recommend_tool_binding("camera")
        self.assertEqual(recommendation["recommended"], "com.whatsapp")

        # Stats
        stats = kg.get_stats()
        self.assertEqual(stats["total_nodes"], 2)
        self.assertEqual(stats["relationships"], 1)


if __name__ == "__main__":
    unittest.main()
