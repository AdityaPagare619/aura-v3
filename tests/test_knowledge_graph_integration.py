"""
Integration test: Knowledge Graph with App Discovery and Tool Binding

This test demonstrates the full workflow:
1. App discovery finds apps
2. Knowledge graph maps topology
3. Query engine recommends tools
4. Tool binding uses recommendations
"""

import unittest
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory.knowledge_graph import (
    KnowledgeGraph,
    AppNode,
    QueryEngine,
    RelationshipEdge,
    RelationshipType,
    get_knowledge_graph,
)
from src.addons.discovery import AppDiscovery, AppMetadata, AppCapability, AppCategory


class MockAppDiscovery:
    """Mock app discovery for testing"""

    def __init__(self):
        self._apps = {}

        # Add some mock apps
        self._apps["com.whatsapp"] = type(
            "AppEntry",
            (),
            {
                "id": "com.whatsapp",
                "metadata": AppMetadata(
                    name="WhatsApp",
                    description="Messaging app",
                    capabilities=[
                        AppCapability.CAMERA_ACCESS,
                        AppCapability.VOICE_INPUT,
                    ],
                    category=AppCategory.COMMUNICATION,
                ),
            },
        )()

        self._apps["com.google.android.apps.photos"] = type(
            "AppEntry",
            (),
            {
                "id": "com.google.android.apps.photos",
                "metadata": AppMetadata(
                    name="Google Photos",
                    description="Photo gallery",
                    capabilities=[
                        AppCapability.CAMERA_ACCESS,
                        AppCapability.FILE_ACCESS,
                    ],
                    category=AppCategory.MEDIA,
                ),
            },
        )()

        self._apps["com.spotify"] = type(
            "AppEntry",
            (),
            {
                "id": "com.spotify",
                "metadata": AppMetadata(
                    name="Spotify",
                    description="Music streaming",
                    capabilities=[AppCapability.VOICE_OUTPUT],
                    category=AppCategory.MEDIA,
                ),
            },
        )()


class TestKnowledgeGraphToolBindingIntegration(unittest.TestCase):
    """Test knowledge graph integration with tool binding"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "integration_test.db")
        self.kg = KnowledgeGraph(self.db_path)
        self.query_engine = QueryEngine(self.kg)

    def tearDown(self):
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _map_from_app_discovery(self, mock_discovery):
        """Helper to map apps from mock discovery"""
        count = 0
        for app_id, app_entry in mock_discovery._apps.items():
            node = AppNode(
                id=app_id,
                name=app_entry.metadata.name,
                package_name=app_id,
                capabilities=[cap.value for cap in app_entry.metadata.capabilities],
                category=app_entry.metadata.category.value,
                description=app_entry.metadata.description,
            )
            self.kg.add_node(node)
            count += 1
        return count

    def test_app_discovery_to_knowledge_graph(self):
        """Test mapping app discovery to knowledge graph"""
        mock_discovery = MockAppDiscovery()

        # Map apps manually (simulating initialize_from_app_discovery)
        count = self._map_from_app_discovery(mock_discovery)

        self.assertEqual(count, 3)
        self.assertEqual(len(self.kg._node_cache), 3)

    def test_capability_based_recommendation(self):
        """Test getting tool binding recommendations by capability"""
        mock_discovery = MockAppDiscovery()
        self._map_from_app_discovery(mock_discovery)

        # Simulate usage - WhatsApp is most used
        whatsapp = self.kg.get_node("com.whatsapp")
        self.assertIsNotNone(whatsapp)
        whatsapp.use_count = 50
        whatsapp.is_running = True
        self.kg.add_node(whatsapp)

        # Get recommendation for camera capability
        recommendation = self.query_engine.recommend_tool_binding("camera_access")

        self.assertIsNotNone(recommendation["recommended"])
        self.assertEqual(
            recommendation["recommended"], "com.whatsapp"
        )  # Most used app with camera
        self.assertGreaterEqual(recommendation["confidence"], 0.5)

    def test_finding_alternatives(self):
        """Test finding alternative apps"""
        mock_discovery = MockAppDiscovery()
        self._map_from_app_discovery(mock_discovery)

        # Add a relationship
        self.kg.add_relationship(
            RelationshipEdge(
                id="rel_1",
                source_id="com.whatsapp",
                target_id="com.google.android.apps.photos",
                relationship_type=RelationshipType.DATA_SHARE,
                weight=0.8,
            )
        )

        # Find alternatives
        alternatives = self.query_engine.find_alternatives(
            "com.whatsapp", RelationshipType.DATA_SHARE
        )

        self.assertTrue(len(alternatives) > 0)

    def test_data_flow_analysis(self):
        """Test data flow between apps"""
        mock_discovery = MockAppDiscovery()
        self._map_from_app_discovery(mock_discovery)

        # WhatsApp can provide photos
        whatsapp = self.kg.get_node("com.whatsapp")
        self.assertIsNotNone(whatsapp)
        whatsapp.export_data_schemas = ["photos"]
        whatsapp.data_types = ["photos"]  # Add to data_types for index
        self.kg.add_node(whatsapp)

        # Photos can import photos
        photos = self.kg.get_node("com.google.android.apps.photos")
        self.assertIsNotNone(photos)
        photos.import_data_schemas = ["photos"]
        photos.data_types = ["photos"]  # Add to data_types for index
        self.kg.add_node(photos)

        # Find data handlers
        handlers = self.query_engine.find_data_handlers("photos")

        # Both should handle photos
        photo_ids = [app.id for app in handlers["readers"]]
        self.assertIn("com.google.android.apps.photos", photo_ids)


if __name__ == "__main__":
    unittest.main()
