from setuptools import setup, find_packages

setup(
    name='aichamptools',
    version='0.1',
    packages=find_packages(),
    description='Various useful tools to work with LLMs. Prompt Engineering, experiments, etc',
    author='Tony AI Champ',
    author_email='tony@aicha.mp',
    url='https://aicha.mp',  # If you have a url
    install_requires=[
        'numpy',
        'pandas',
        'sqlalchemy',
        'mistralai',
        'openai'
    ],
)