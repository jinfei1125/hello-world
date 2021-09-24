# Running Jupyter Notebook on WSL2

1. First set up WSL2 and download Ubuntu distribution following steps on: [https://docs.microsoft.com/en-us/windows/wsl/install-win10](https://docs.microsoft.com/en-us/windows/wsl/install-win10)
Manual Installation is recommended
2. Folders and Files can be accessed at: `\\wsl$`
3. Download Anaconda Distribution via:
	```
	wget https://repo.continuum.io/archive/Anaconda3-2020.07-Linux-x86_64.sh
	bash Anaconda3-2020.07-Linux-x86_64.sh
	```
4. Install Jupyter Notebook

	```
	sudo apt-get update
	sudo apt install jupyter-core
	sudo apt install jupyter
	```

5. Open jupyter notebook by `jupyter notebook --no-browser`


Reference: [http://mcb112.org/w00/w00-section.html](http://mcb112.org/w00/w00-section.html)