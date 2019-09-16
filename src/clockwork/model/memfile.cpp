#include "clockwork/model/memfile.h"
#include "dmlc/logging.h"
#include <unistd.h>
#include <sys/syscall.h>
#include <iostream>
#include <sys/stat.h> 
#include <fcntl.h>

namespace clockwork {

int make_memfd() {
	int fd = syscall(__NR_memfd_create, std::string("").c_str(), 0);
	if (fd < 0) {
		perror("memfd_create");
		CHECK(fd < 0) << "ModelCodeReader could not create memfile";
	}
	return fd;
}

std::string memfd_filename(int memfd) {
	std::stringstream ss;
	ss << "/proc/" << getpid() << "/fd/" << memfd;
	return ss.str();
}

Memfile Memfile::readFrom(const std::string &filename) {
	int memfd = make_memfd();
	std::string memfilename = memfd_filename(memfd);
    std::ofstream dst(memfilename,   std::ios::binary);

	std::ifstream src(filename, std::ios::binary);
    dst << src.rdbuf();

    src.close();
    dst.close();

    return Memfile(memfd, memfilename);
}

Memfile Memfile::fromBuffer(const void *buf, size_t len)
{
    int memfd = make_memfd();
    std::string memfilename = memfd_filename(memfd);
    std::ofstream dst(memfilename, std::ios::binary);

    dst.write(reinterpret_cast<const char *>(buf), len);
    dst.close();

    return Memfile(memfd, memfilename);
}

MemfileReader::MemfileReader(const Memfile &memfile) : 
	memfile(memfile), dst(memfile.filename, std::ios::binary) {
}

unsigned MemfileReader::readsome(std::istream stream, unsigned amount) {
	char buf[amount];
	int amountRead = stream.readsome(buf, amount);
	dst.write(buf, amountRead);
	return amountRead;
}
	
}