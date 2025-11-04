import sys

import infinicore


def use_ntops():
    import ntops

    return _TemporaryAttributes(
        tuple(
            (f"infinicore.{op_name}", getattr(ntops.torch, op_name))
            for op_name in ntops.torch.__all__
        ) 
        + tuple(
            (f"ntops.torch.{op_name}.__globals__['torch']", infinicore)
            for op_name in ntops.torch.__all__
        )
    )


class _TemporaryAttributes:
    def __init__(self, attribute_mappings):
        self._attribute_mappings = attribute_mappings

        self._original_values = {}

    def __enter__(self):
        for attr_path, new_value in self._attribute_mappings:
            parent, attr_name, is_dict_key = self._resolve_path(attr_path)

            try:
                if is_dict_key:
                    self._original_values[attr_path] = parent.__globals__[attr_name]
                else:
                    self._original_values[attr_path] = getattr(parent, attr_name)
            except (AttributeError, KeyError):
                pass

            if is_dict_key:
                parent.__globals__[attr_name] = new_value
            else:
                setattr(parent, attr_name, new_value)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for attr_path, _ in self._attribute_mappings:
            parent, attr_name, is_dict_key = self._resolve_path(attr_path)

            if attr_path in self._original_values:
                original_value = self._original_values[attr_path]
                if is_dict_key:
                    parent.__globals__[attr_name] = original_value
                else:
                    setattr(parent, attr_name, original_value)
            else:
                if is_dict_key:
                    if attr_name in parent.__globals__.keys():
                        del parent.__globals__[attr_name]
                else:
                    if parent is not None and attr_name is not None:
                        delattr(parent, attr_name)

    @staticmethod
    def _resolve_path(path):
        is_dict_key = False
        dict_key_match = None
        
        if path.endswith("']"):
            try:
                start_index = path.rindex("['")
                end_index = path.rindex("']")
                
                if start_index > 0 and end_index == len(path) - 2:
                    is_dict_key = True
                    dict_key_match = path[start_index + 2 : end_index]
                    path = path[:start_index]
            except ValueError:
                pass

        *parent_parts, attr_name = path.split(".")

        curr = sys.modules[parent_parts[0]]

        for part in parent_parts[1:]:
            curr = getattr(curr, part)
        
        parent = curr 

        if is_dict_key:
            return parent, dict_key_match, True
        else:
            return parent, attr_name, False
