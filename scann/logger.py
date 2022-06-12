import time
import colorful as cf


class Logger:
    logging_enabled = True
    palette = {
        "red": "#F75590",
        "blue": "#3DB1F5",
        "white": "#FFFFFF",
        "green": "#9EE493",
        "yellow": "#FFF689",
        "black": "#000000"
    }
    @staticmethod
    def log(logger, msg, *args):
        cf.use_palette(Logger.palette)
        if Logger.logging_enabled:
            print(
                (
                ("[{c.bold}{c.white}%s{c.reset} - {c.green}%s{c.reset} - {c.blue}%s{c.reset}] " + msg)
                %
                ((type(logger).__name__, hex(id(logger)).replace("0x", ""), time.strftime("%Y-%m-%d %H:%M:%S")) + args)
                ).format(c=cf)
            )