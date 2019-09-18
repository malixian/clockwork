
#ifndef _CLOCKWORK_MEMFILE_H_
#define _CLOCKWORK_MEMFILE_H_

#include <string>
#include <istream>
#include <fstream>

namespace clockwork {

/** An in-memory file */
class Memfile {
public:
	const int memfd;
	const std::string filename;

	Memfile(const int &memfd, const std::string &filename) :
		memfd(memfd), filename(filename) {}

	// Copy another file into a memfile
	static Memfile readFrom(const std::string &filename);

};

/** Used for reading data from a stream into a memfile */
class MemfileReader {
public:
	const Memfile &memfile;
	MemfileReader(const Memfile &memfile);

private:
	std::ofstream dst;

	unsigned readsome(std::istream stream, unsigned amount);

};

}

#endif