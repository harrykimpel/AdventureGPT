"""
Create a shell through which the LLM agents can play Adventure.

Copyright 2010-2015 Brandon Rhodes.
Copyright 2023 Lily Hughes-Robinson.

Licensed as free software under the
Apache License, Version 2.0 as detailed in the accompanying README.txt.
"""
import newrelic.agent
import argparse
from adventuregpt.loop import Loop

# initialize the New Relic Python agent
newrelic.agent.initialize('newrelic.ini')

application = newrelic.agent.register_application(
    timeout=5)  # force New Relic agent registration [RLF]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="AdventureGPT",
        description="The game ADVENTURE played by ChatGPT"
    )
    parser.add_argument("-w", "--walkthrough_path")
    parser.add_argument("-o", "--output_path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    try:
        game_loop = Loop(args.walkthrough_path, args.output_path, args.verbose)
        game_loop.loop()
    except EOFError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        newrelic.agent.shutdown_agent(timeout=2.5)  # shutdown New Relic agent [RLF]
        
        game_loop.dump_history()
