class BaseTool:
    def __init__(self, args):
        self.args = args
        self.args_base = self.args["base"]
        self.args_rag = self.args["rag"]
        self.args_prompt = self.args["prompt"]

        self.log_level = self.args_base["log_level"]

    def level_print(self, input_str, level):
        if level <= self.log_level:
            print(input_str)


