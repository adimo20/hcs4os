from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from requests import Session
from requests.auth import HTTPBasicAuth
from zeep import Client
from zeep.helpers import serialize_object
from zeep.transports import Transport

WSDL = Path(__file__).with_name("klassservice_fixed.wsdl")
BERLIN = ZoneInfo("Europe/Berlin")


class KlassServerModule:

    def __init__(
        self,
        username: str = "anonymous",
        password: str = "anonymous",
        language: str = "DE",
    ) -> None:
        self.language = language

        session = Session()
        session.auth = HTTPBasicAuth(username, password)

        self.client = Client(
            str(WSDL),
            transport=Transport(session=session, timeout=30),
        )

    def ping(self) -> str:
        return self.client.service.ping()

    def search_item(self, query: str, classification: str = "WZ2008") -> list[dict]:
        """Free-text search. Returns a list of {"key": ..., "title": ...}."""
        response = self.client.service.searchItem(
            self.language, classification, datetime.now(BERLIN), query
        )
        return serialize_object(response) or []


if __name__ == "__main__":
    klass = KlassServerModule()
    print("ping:", klass.ping())
    print(klass.search_item("Systemadmin", classification="kldb2010"))