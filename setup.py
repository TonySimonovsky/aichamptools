from setuptools import setup, find_packages

setup(
    name='aichamptools',
    version='0.2.1',
    description='A library for LLMOps including connectors to various LLMs and assessment tools',
    author='Tony AI Champ',
    author_email='tony@aicha.mp',
    packages=find_packages(include=['aichamptools', 'aichamptools.*']),
    install_requires=[
        'requests',
        'pydantic',
        'numpy',
        'tiktoken',
        'openai',
        'pydub',
        'mistralai',
        'anthropic',
        'sqlalchemy',
        'replicate'
    ],
)
