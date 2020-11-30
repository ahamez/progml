# Programming Machine Learning

This is the source code for Paolo Perrotta's [Programming Machine Learning](http://www.pragmaticprogrammer.com/titles/pplearn) book.

To run this code, you need [Python 3](https://www.python.org/downloads/) (or greater) and a few libraries. You can install the libraries via Python's built-in package manager _pip_, or via the more sophisticated _Conda_ package manager. Let's look at both.


### Installing with pip

If you have Python 3, then you should also have pip. You can install the libraries straight away:

    pip3 install numpy==1.15.2
    pip3 install matplotlib==3.1.2
    pip3 install seaborn==0.9.0
    pip3 install scikit-learn==0.22.1
    pip3 install keras==2.2.4

If you prefer to run the code in a Jupyter Notebook, then you also need Jupyter:

    pip3 install jupyter==1.0.0


### Installing with Conda

If you want to control the visibility of your libraries, consider installing them with the [Conda](https://conda.io) package manager. Compared to pip, Conda helps you keep Python environments tidy and isolated. On the minus side, Conda doesn't always play nice with other package managers, such as Homebrew. I prefer Conda over pip, but your mileage may vary.

If you opt for Conda, consider downloading [Miniconda](https://docs.conda.io/en/latest/miniconda.html), the minimal distribution, because the complete distribution is a large install.

Here is how you create a new Python 3 environment with Conda:

    conda create --name=machinelearning python=3

Now you can make `machinelearning` the current active environment:

    source activate machinelearning

(In modern versions of Conda, you can also use `conda activate` instead of `source activate`.)

Next step, you can install libraries in the active environment:

    conda install numpy=1.15.2
    conda install matplotlib=3.1.2
    conda install seaborn=0.9.0
    conda install scikit-learn=0.22.1
    conda install keras=2.2.4
    conda install jupyter==1.0.0

The libraries will stay visible as long as the environment is active. Once you deactivate the environment with `source deactivate`, or close the terminal, the libraries are gone. To re-activate the environment and get back the libraries, use `source activate machinelearning` again.

Happy hacking!


## Copyright

Excerpted from "Programming Machine Learning", published by The Pragmatic Bookshelf.
Copyrights apply to this code. It may not be used to create training material, courses, books, articles, and the like. Contact us if you are in doubt.
We make no guarantees that this code is fit for any purpose.
Visit http://www.pragmaticprogrammer.com/titles/pplearn for more book information.
