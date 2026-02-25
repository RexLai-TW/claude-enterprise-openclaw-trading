"""Tests for MCP bridge"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.mcp_bridge import MCPBridge, MCPTool, create_mcp_bridge
from src.data.market_data import create_market_data_connector


class TestMCPBridge:
    def test_bridge_creation(self):
        bridge = MCPBridge()
        assert bridge is not None

    def test_tools_registered(self):
        bridge = MCPBridge()
        result = bridge._handle_list_tools()
        assert "tools" in result
        assert len(result["tools"]) > 0

    def test_tool_structure(self):
        bridge = MCPBridge()
        result = bridge._handle_list_tools()
        for tool in result["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool


class TestMarketDataConnector:
    def test_create_connector(self):
        connector = create_market_data_connector()
        assert connector is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
