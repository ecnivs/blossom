from .base import Plugin


class testplugin(Plugin):
    description = "this is a test plugin"

    @staticmethod
    def run():
        print("test plugin works")
