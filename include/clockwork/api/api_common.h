#ifndef _CLOCKWORK_API_COMMON_H_
#define _CLOCKWORK_API_COMMON_H_

#include <functional>
#include <string>

const int clockworkSuccess = 0;
const int clockworkError = 1;

namespace clockwork {
namespace api {

struct RequestHeader {
	int user_id;
	int user_request_id;
};

struct ResponseHeader {
	int user_request_id;
	int status;
	std::string message;
};

}
}

#endif