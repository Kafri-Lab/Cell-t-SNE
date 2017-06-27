# WORK THE PROGRESS
# TODO: implement the following functionality
#    python kafri-tsne.py --input-images barcode_images/ --perplexity 20 --output barcode_canvas.jpg
import click

@click.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name',
              help='The person to greet.')
def hello(count, name):
    """Simple program that greets NAME for a total of COUNT times."""
    for x in range(count):
        click.echo('Hello %s!' % name)

if __name__ == '__main__':
    hello()
