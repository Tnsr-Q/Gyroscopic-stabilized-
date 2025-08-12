"""Event bus framework for loose coupling between components."""

from typing import Dict, List, Callable, Any
import logging

logger = logging.getLogger(__name__)

class EventBus:
    """Simple event bus for component communication."""
    
    def __init__(self):
        self.listeners: Dict[str, List[Callable]] = {}
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to an event type."""
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)
        logger.debug(f"Subscribed to event: {event_type}")
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from an event type."""
        if event_type in self.listeners:
            try:
                self.listeners[event_type].remove(callback)
                logger.debug(f"Unsubscribed from event: {event_type}")
            except ValueError:
                logger.warning(f"Callback not found for event: {event_type}")
    
    def publish(self, event_type: str, data: Any = None):
        """Publish an event to all subscribers."""
        if event_type in self.listeners:
            for callback in self.listeners[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in event callback for {event_type}: {e}")
        logger.debug(f"Published event: {event_type}")
    
    def clear_listeners(self, event_type: str = None):
        """Clear listeners for a specific event type or all events."""
        if event_type:
            self.listeners.pop(event_type, None)
        else:
            self.listeners.clear()
    
    def get_listener_count(self, event_type: str) -> int:
        """Get number of listeners for an event type."""
        return len(self.listeners.get(event_type, []))