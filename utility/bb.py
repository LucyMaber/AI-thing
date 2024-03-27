import curses
from pygments import highlight
from pygments.lexers import SqlLexer
from pygments.formatters import TerminalFormatter


class CustomMultilineInput:
    def __init__(self, prompt="SQLite> "):
        self.text = ""
        self.terminator = ";"
        self.terminator_index = 0
        self.prompt = prompt

        # # Initialize curses
        # self.stdscr = curses.initscr()
        # # Define color pairs (you can customize these)
        # curses.start_color()
        # curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
        # curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
        # curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        # curses.init_pair(4, curses.COLOR_BLUE, curses.COLOR_BLACK)
        # curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        # curses.init_pair(6, curses.COLOR_CYAN, curses.COLOR_BLACK)
        # curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLACK)
        # curses.cbreak()
        # curses.noecho()

    def highlight_sql(self, text):
        lexer = SqlLexer()
        formatter = TerminalFormatter()
        highlighted_sql = highlight(text, lexer, formatter)
        return highlighted_sql

    def input(self):
        self.stdscr.keypad(True)

        self.stdscr.addstr(self.prompt)
        self.stdscr.refresh()

        while True:
            char = self.stdscr.getch()

            # Handle special keys
            if chr(char) == self.terminator:
                self.text += self.terminator
                break
            elif char == curses.KEY_BACKSPACE or char == 127:
                if self.text:
                    self.text = self.text[:-1]
                    self.stdscr.addch(char)
                    self.stdscr.refresh()
            else:
                self.text += chr(char)
                self.stdscr.addch(char)

            # Highlight SQL as the user types
            highlighted_sql = self.highlight_sql(self.text)

            self.stdscr.clear()
            self.stdscr.addstr(self.prompt + highlighted_sql)
            self.stdscr
            self.stdscr.refresh()
            self.stdscr.refresh()

        # curses.endwin()
        return self.text


if __name__ == "__main__":
    input_handler = CustomMultilineInput()
    user_input = input_handler.input()
