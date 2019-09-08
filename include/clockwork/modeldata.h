
#ifndef CLOCKWORK_MODELDATA_H_
#define CLOCKWORK_MODELDATA_H_

#include <string>
#include <unistd.h>
#include <sys/syscall.h>
#include <iostream>
#include <sys/stat.h> 
#include <fcntl.h>
#include <istream>
#include <sstream>
#include <fstream>
#include "dmlc/logging.h"

namespace clockwork {
namespace binary {

/**
Represents in-memory model data, with minimal amount of model deserialization or construction.

They key things that ModelData DOES do are the following:

(1) Part of ModelData is a shared object containing model code.  dlopen only operates on files,
so the model code is stored in a memory mapped file, to avoid later copying. (TODO: is this the only way?)

(2) Model weights are immediately stored in CUDA-pinned memory

(3) Model metadata is just a blob stored in memory

ModelData is designed so that it can be read off the network or disk with minimal decoding,
and similarly written back to the network or disk with minimal encoding.
*/
class ModelData {
private:


};

class ModelCodeReader {
public:
	int memfd;
	std::string memfile;

private:
	std::ofstream memstrm;
	unsigned remaining;

	ModelCodeReader(unsigned remaining, const int &memfd, const std::string &memfile) : 
		remaining(remaining), memfd(memfd), memfile(memfile), memstrm(memfile, std::ios::binary) {
	}
public:

	static ModelCodeReader create(uint size) {
		int fd = syscall(__NR_memfd_create, std::string("").c_str(), 0);
		if (fd < 0) {
			perror("memfd_create");
			CHECK(fd < 0) << "ModelCodeReader could not create memfile";
		}

		std::stringstream ss;
		ss << "/proc/" << getpid() << "/fd/" << fd;
		return ModelCodeReader(size, fd, ss.str());
	}

	/*
	Read model code from the provided inputstream, and pipe it into the memfile.

	Reads up to remaining-many bytes from the stream.  Will not block if data
	is not available, so multiple calls to this method might be required.

	Returns true if finished reading; false otherwise.

	TODO: more efficient / less buffering
	*/
	bool readFrom(std::istream &istream) {
		if (remaining == 0) return true;

		int bufSize = 1024;
		char buf[bufSize];
		int amountRead, amountToRead;
		do {
			amountToRead = remaining < bufSize ? remaining : bufSize;
			amountRead = istream.readsome(buf, amountToRead);
			remaining -= amountRead;
			memstrm.write(buf, amountRead);
		} while (remaining > 0 && amountRead == amountToRead);

		if (remaining == 0) {
			memstrm.close();
			return true;
		}

		return false;
	}






};

class ModelWeightsReader {

};

class ModelMetadataReader {

};

class ModelDataReader {
private:
	

};

}
}

#endif