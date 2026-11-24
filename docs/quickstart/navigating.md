# Navigating a classification system

```python
from hcs4os.classification_system import get_classification_system

sea = get_classification_system("SEA_2021")

sea.get_root_categories()[:2]
# [{'code': '00', 'description': 'Einnahmen der privaten Haushalte'},
#  {'code': '01', 'description': 'Nahrungsmittel und alkoholfreie Getränke'}]

sea.get_children("0111")
# [Code(code='0111 1', description='Getreide', level=4, ...), ...]

sea.get_code("0111 1").to_dict()
sea.get_parent("0111 1")          # -> Code(code='0111', description='Getreide und …')
sea.get_code_trace("0111 1")
# [('01', 'Nahrungsmittel und alkoholfreie Getränke'),
#  ('011', 'Nahrungsmittel'),
#  ('0111', 'Getreide und Getreideerzeugnisse'),
#  ('0111 1', 'Getreide')]
```
