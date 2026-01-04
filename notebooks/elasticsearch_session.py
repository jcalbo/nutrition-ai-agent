"""
Implementación de sesión usando Elasticsearch como backend para almacenar
el historial de conversación de agentes.

Autor: Sistema de IA
Fecha: 2026-01-04
"""

from __future__ import annotations
import json
from datetime import datetime
from typing import TYPE_CHECKING
from elasticsearch import AsyncElasticsearch
from agents.memory.session import SessionABC

if TYPE_CHECKING:
    from agents.items import TResponseInputItem


class ElasticsearchSession(SessionABC):
    """
    Implementación de sesión usando Elasticsearch como backend.
    
    Esta clase almacena el historial de conversación en un índice de Elasticsearch,
    permitiendo persistencia distribuida y búsquedas avanzadas del historial.
    
    Atributos:
        session_id: Identificador único de la conversación
        index_name: Nombre del índice de Elasticsearch donde se almacenan las sesiones
        es_client: Cliente asíncrono de Elasticsearch
    """

    def __init__(
        self,
        session_id: str,
        es_host: str = "localhost:9200",
        index_name: str = "agent_sessions",
        username: str | None = None,
        password: str | None = None,
    ):
        """
        Inicializa la sesión de Elasticsearch.

        Args:
            session_id: Identificador único para la conversación
            es_host: Host y puerto de Elasticsearch (ej: "localhost:9200")
            index_name: Nombre del índice de Elasticsearch
            username: Usuario para autenticación (opcional)
            password: Contraseña para autenticación (opcional)
        """
        self.session_id = session_id
        self.index_name = index_name
        
        # Configurar cliente de Elasticsearch
        es_config = {"hosts": [f"http://{es_host}"]}
        if username and password:
            es_config["basic_auth"] = (username, password)
        
        self.es_client = AsyncElasticsearch(**es_config)
        
    async def _ensure_index_exists(self):
        """
        Crea el índice de Elasticsearch si no existe.
        
        Define el mapping con:
        - session_id: identificador de la sesión (keyword)
        - timestamp: momento de creación del mensaje
        - item_data: datos del mensaje (objeto no analizado)
        - sequence: número de secuencia para ordenar mensajes
        """
        if not await self.es_client.indices.exists(index=self.index_name):
            await self.es_client.indices.create(
                index=self.index_name,
                body={
                    "mappings": {
                        "properties": {
                            "session_id": {"type": "keyword"},
                            "timestamp": {"type": "date"},
                            "item_data": {"type": "text", "index": False},  # String JSON no indexado
                            "sequence": {"type": "long"}
                        }
                    }
                }
            )

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        """
        Recupera el historial de conversación de Elasticsearch.
        
        Los items se retornan en orden cronológico (del más antiguo al más reciente).

        Args:
            limit: Número máximo de items a recuperar. Si es None, recupera todos.

        Returns:
            Lista de items del historial de conversación
        """
        await self._ensure_index_exists()
        
        # Construir query de búsqueda
        query = {
            "query": {
                "term": {"session_id": self.session_id}
            },
            "sort": [{"sequence": {"order": "asc"}}]
        }
        
        if limit:
            query["size"] = limit
            # Si hay límite, obtener los últimos N items
            query["sort"] = [{"sequence": {"order": "desc"}}]
        else:
            query["size"] = 10000  # Máximo de Elasticsearch
        
        # Ejecutar búsqueda
        response = await self.es_client.search(
            index=self.index_name,
            body=query
        )
        
        # Deserializar items desde JSON strings
        items = []
        for hit in response["hits"]["hits"]:
            item_json = hit["_source"]["item_data"]
            try:
                item = json.loads(item_json)
                items.append(item)
            except (json.JSONDecodeError, TypeError):
                # Skip invalid JSON entries
                continue
        
        # Si hay límite, invertir para mantener orden cronológico
        if limit:
            items.reverse()
        
        return items

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        """
        Agrega nuevos items al historial en Elasticsearch.
        
        Usa bulk insert para eficiencia cuando se agregan múltiples items.

        Args:
            items: Lista de items a agregar al historial
        """
        # DEBUG: Imprimir cuando se llama este método
        print(f"[DEBUG] add_items() llamado con {len(items)} items")
        
        if not items:
            print("[DEBUG] Lista de items vacía, retornando sin hacer nada")
            return
            
        await self._ensure_index_exists()
        
        # Obtener el último número de secuencia para mantener orden
        last_seq = await self._get_last_sequence()
        print(f"[DEBUG] Última secuencia: {last_seq}")
        
        # Preparar documentos para bulk insert
        bulk_body = []
        for idx, item in enumerate(items):
            sequence = last_seq + idx + 1
            
            # DEBUG: Imprimir tipo del item
            print(f"[DEBUG] Item {idx}: tipo={type(item)}")
            
            # Serializar el item como JSON string (igual que SQLiteSession)
            try:
                item_json = json.dumps(item, default=str)
                print(f"[DEBUG] Item serializado exitosamente, longitud={len(item_json)}")
            except Exception as e:
                print(f"[DEBUG ERROR] Error serializando item {idx}: {e}")
                continue
            
            # Crear acción de indexación
            bulk_body.append({
                "index": {
                    "_index": self.index_name,
                }
            })
            
            # Documento con los datos del mensaje
            bulk_body.append({
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),  # Timestamp en formato ISO
                "sequence": sequence,
                "item_data": item_json  # Guardar como string JSON
            })
        
        # Ejecutar bulk insert con refresh inmediato
        if bulk_body:
            print(f"[DEBUG] Ejecutando bulk insert con {len(bulk_body)//2} documentos")
            try:
                result = await self.es_client.bulk(body=bulk_body, refresh=True)
                print(f"[DEBUG] Bulk insert exitoso: errors={result.get('errors', False)}")
                if result.get('errors'):
                    print(f"[DEBUG ERROR] Detalles de errores: {result.get('items', [])[:3]}")
            except Exception as e:
                print(f"[DEBUG ERROR] Error en bulk insert: {e}")

    async def pop_item(self) -> TResponseInputItem | None:
        """
        Elimina y retorna el item más reciente de la sesión.
        
        Útil para deshacer la última acción en una conversación.

        Returns:
            El item más reciente o None si la sesión está vacía
        """
        await self._ensure_index_exists()
        
        # Buscar el último item por número de secuencia
        response = await self.es_client.search(
            index=self.index_name,
            body={
                "query": {"term": {"session_id": self.session_id}},
                "sort": [{"sequence": {"order": "desc"}}],
                "size": 1
            }
        )
        
        if not response["hits"]["hits"]:
            return None
        
        hit = response["hits"]["hits"][0]
        item_json = hit["_source"]["item_data"]
        doc_id = hit["_id"]
        
        # Deserializar el item
        try:
            item = json.loads(item_json)
        except (json.JSONDecodeError, TypeError):
            return None
        
        # Eliminar el documento de Elasticsearch
        await self.es_client.delete(
            index=self.index_name,
            id=doc_id,
            refresh=True
        )
        
        return item

    async def clear_session(self) -> None:
        """
        Elimina todos los items de esta sesión.
        
        Usa delete_by_query para eliminar eficientemente todos los documentos
        asociados al session_id.
        """
        await self._ensure_index_exists()
        
        # Eliminar todos los documentos de esta sesión
        await self.es_client.delete_by_query(
            index=self.index_name,
            body={
                "query": {
                    "term": {"session_id": self.session_id}
                }
            },
            refresh=True
        )

    async def _get_last_sequence(self) -> int:
        """
        Obtiene el último número de secuencia para esta sesión.
        
        Usado internamente para asignar números de secuencia consecutivos
        a nuevos mensajes.

        Returns:
            Último número de secuencia, o 0 si la sesión está vacía
        """
        response = await self.es_client.search(
            index=self.index_name,
            body={
                "query": {"term": {"session_id": self.session_id}},
                "sort": [{"sequence": {"order": "desc"}}],
                "size": 1
            }
        )
        
        if response["hits"]["hits"]:
            return response["hits"]["hits"][0]["_source"]["sequence"]
        return 0
    
    async def close(self):
        """
        Cierra la conexión con Elasticsearch.
        
        Debe llamarse al finalizar para liberar recursos.
        """
        await self.es_client.close()

