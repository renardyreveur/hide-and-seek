/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
"use strict";

var $protobuf = require("protobufjs/minimal");

// Common aliases
var $Reader = $protobuf.Reader, $Writer = $protobuf.Writer, $util = $protobuf.util;

// Exported root namespace
var $root = $protobuf.roots["default"] || ($protobuf.roots["default"] = {});

$root.game = (function() {

    /**
     * Namespace game.
     * @exports game
     * @namespace
     */
    var game = {};

    game.Point = (function() {

        /**
         * Properties of a Point.
         * @memberof game
         * @interface IPoint
         * @property {number|null} [x] Point x
         * @property {number|null} [y] Point y
         */

        /**
         * Constructs a new Point.
         * @memberof game
         * @classdesc Represents a Point.
         * @implements IPoint
         * @constructor
         * @param {game.IPoint=} [properties] Properties to set
         */
        function Point(properties) {
            if (properties)
                for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * Point x.
         * @member {number} x
         * @memberof game.Point
         * @instance
         */
        Point.prototype.x = 0;

        /**
         * Point y.
         * @member {number} y
         * @memberof game.Point
         * @instance
         */
        Point.prototype.y = 0;

        /**
         * Creates a new Point instance using the specified properties.
         * @function create
         * @memberof game.Point
         * @static
         * @param {game.IPoint=} [properties] Properties to set
         * @returns {game.Point} Point instance
         */
        Point.create = function create(properties) {
            return new Point(properties);
        };

        /**
         * Encodes the specified Point message. Does not implicitly {@link game.Point.verify|verify} messages.
         * @function encode
         * @memberof game.Point
         * @static
         * @param {game.IPoint} message Point message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Point.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.x != null && Object.hasOwnProperty.call(message, "x"))
                writer.uint32(/* id 1, wireType 0 =*/8).int32(message.x);
            if (message.y != null && Object.hasOwnProperty.call(message, "y"))
                writer.uint32(/* id 2, wireType 0 =*/16).int32(message.y);
            return writer;
        };

        /**
         * Encodes the specified Point message, length delimited. Does not implicitly {@link game.Point.verify|verify} messages.
         * @function encodeDelimited
         * @memberof game.Point
         * @static
         * @param {game.IPoint} message Point message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Point.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a Point message from the specified reader or buffer.
         * @function decode
         * @memberof game.Point
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {game.Point} Point
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        Point.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            var end = length === undefined ? reader.len : reader.pos + length, message = new $root.game.Point();
            while (reader.pos < end) {
                var tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.x = reader.int32();
                    break;
                case 2:
                    message.y = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a Point message from the specified reader or buffer, length delimited.
         * @function decodeDelimited
         * @memberof game.Point
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {game.Point} Point
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        Point.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a Point message.
         * @function verify
         * @memberof game.Point
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        Point.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.x != null && message.hasOwnProperty("x"))
                if (!$util.isInteger(message.x))
                    return "x: integer expected";
            if (message.y != null && message.hasOwnProperty("y"))
                if (!$util.isInteger(message.y))
                    return "y: integer expected";
            return null;
        };

        /**
         * Creates a Point message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof game.Point
         * @static
         * @param {Object.<string,*>} object Plain object
         * @returns {game.Point} Point
         */
        Point.fromObject = function fromObject(object) {
            if (object instanceof $root.game.Point)
                return object;
            var message = new $root.game.Point();
            if (object.x != null)
                message.x = object.x | 0;
            if (object.y != null)
                message.y = object.y | 0;
            return message;
        };

        /**
         * Creates a plain object from a Point message. Also converts values to other types if specified.
         * @function toObject
         * @memberof game.Point
         * @static
         * @param {game.Point} message Point
         * @param {$protobuf.IConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        Point.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            var object = {};
            if (options.defaults) {
                object.x = 0;
                object.y = 0;
            }
            if (message.x != null && message.hasOwnProperty("x"))
                object.x = message.x;
            if (message.y != null && message.hasOwnProperty("y"))
                object.y = message.y;
            return object;
        };

        /**
         * Converts this Point to JSON.
         * @function toJSON
         * @memberof game.Point
         * @instance
         * @returns {Object.<string,*>} JSON object
         */
        Point.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return Point;
    })();

    game.Agent = (function() {

        /**
         * Properties of an Agent.
         * @memberof game
         * @interface IAgent
         * @property {number|null} [agentClass] Agent agentClass
         * @property {game.IPoint|null} [location] Agent location
         */

        /**
         * Constructs a new Agent.
         * @memberof game
         * @classdesc Represents an Agent.
         * @implements IAgent
         * @constructor
         * @param {game.IAgent=} [properties] Properties to set
         */
        function Agent(properties) {
            if (properties)
                for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * Agent agentClass.
         * @member {number} agentClass
         * @memberof game.Agent
         * @instance
         */
        Agent.prototype.agentClass = 0;

        /**
         * Agent location.
         * @member {game.IPoint|null|undefined} location
         * @memberof game.Agent
         * @instance
         */
        Agent.prototype.location = null;

        /**
         * Creates a new Agent instance using the specified properties.
         * @function create
         * @memberof game.Agent
         * @static
         * @param {game.IAgent=} [properties] Properties to set
         * @returns {game.Agent} Agent instance
         */
        Agent.create = function create(properties) {
            return new Agent(properties);
        };

        /**
         * Encodes the specified Agent message. Does not implicitly {@link game.Agent.verify|verify} messages.
         * @function encode
         * @memberof game.Agent
         * @static
         * @param {game.IAgent} message Agent message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Agent.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.agentClass != null && Object.hasOwnProperty.call(message, "agentClass"))
                writer.uint32(/* id 1, wireType 0 =*/8).int32(message.agentClass);
            if (message.location != null && Object.hasOwnProperty.call(message, "location"))
                $root.game.Point.encode(message.location, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified Agent message, length delimited. Does not implicitly {@link game.Agent.verify|verify} messages.
         * @function encodeDelimited
         * @memberof game.Agent
         * @static
         * @param {game.IAgent} message Agent message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Agent.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes an Agent message from the specified reader or buffer.
         * @function decode
         * @memberof game.Agent
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {game.Agent} Agent
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        Agent.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            var end = length === undefined ? reader.len : reader.pos + length, message = new $root.game.Agent();
            while (reader.pos < end) {
                var tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.agentClass = reader.int32();
                    break;
                case 2:
                    message.location = $root.game.Point.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes an Agent message from the specified reader or buffer, length delimited.
         * @function decodeDelimited
         * @memberof game.Agent
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {game.Agent} Agent
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        Agent.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies an Agent message.
         * @function verify
         * @memberof game.Agent
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        Agent.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.agentClass != null && message.hasOwnProperty("agentClass"))
                if (!$util.isInteger(message.agentClass))
                    return "agentClass: integer expected";
            if (message.location != null && message.hasOwnProperty("location")) {
                var error = $root.game.Point.verify(message.location);
                if (error)
                    return "location." + error;
            }
            return null;
        };

        /**
         * Creates an Agent message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof game.Agent
         * @static
         * @param {Object.<string,*>} object Plain object
         * @returns {game.Agent} Agent
         */
        Agent.fromObject = function fromObject(object) {
            if (object instanceof $root.game.Agent)
                return object;
            var message = new $root.game.Agent();
            if (object.agentClass != null)
                message.agentClass = object.agentClass | 0;
            if (object.location != null) {
                if (typeof object.location !== "object")
                    throw TypeError(".game.Agent.location: object expected");
                message.location = $root.game.Point.fromObject(object.location);
            }
            return message;
        };

        /**
         * Creates a plain object from an Agent message. Also converts values to other types if specified.
         * @function toObject
         * @memberof game.Agent
         * @static
         * @param {game.Agent} message Agent
         * @param {$protobuf.IConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        Agent.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            var object = {};
            if (options.defaults) {
                object.agentClass = 0;
                object.location = null;
            }
            if (message.agentClass != null && message.hasOwnProperty("agentClass"))
                object.agentClass = message.agentClass;
            if (message.location != null && message.hasOwnProperty("location"))
                object.location = $root.game.Point.toObject(message.location, options);
            return object;
        };

        /**
         * Converts this Agent to JSON.
         * @function toJSON
         * @memberof game.Agent
         * @instance
         * @returns {Object.<string,*>} JSON object
         */
        Agent.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return Agent;
    })();

    game.Size = (function() {

        /**
         * Properties of a Size.
         * @memberof game
         * @interface ISize
         * @property {number|null} [width] Size width
         * @property {number|null} [height] Size height
         */

        /**
         * Constructs a new Size.
         * @memberof game
         * @classdesc Represents a Size.
         * @implements ISize
         * @constructor
         * @param {game.ISize=} [properties] Properties to set
         */
        function Size(properties) {
            if (properties)
                for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * Size width.
         * @member {number} width
         * @memberof game.Size
         * @instance
         */
        Size.prototype.width = 0;

        /**
         * Size height.
         * @member {number} height
         * @memberof game.Size
         * @instance
         */
        Size.prototype.height = 0;

        /**
         * Creates a new Size instance using the specified properties.
         * @function create
         * @memberof game.Size
         * @static
         * @param {game.ISize=} [properties] Properties to set
         * @returns {game.Size} Size instance
         */
        Size.create = function create(properties) {
            return new Size(properties);
        };

        /**
         * Encodes the specified Size message. Does not implicitly {@link game.Size.verify|verify} messages.
         * @function encode
         * @memberof game.Size
         * @static
         * @param {game.ISize} message Size message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Size.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.width != null && Object.hasOwnProperty.call(message, "width"))
                writer.uint32(/* id 1, wireType 0 =*/8).int32(message.width);
            if (message.height != null && Object.hasOwnProperty.call(message, "height"))
                writer.uint32(/* id 2, wireType 0 =*/16).int32(message.height);
            return writer;
        };

        /**
         * Encodes the specified Size message, length delimited. Does not implicitly {@link game.Size.verify|verify} messages.
         * @function encodeDelimited
         * @memberof game.Size
         * @static
         * @param {game.ISize} message Size message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Size.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a Size message from the specified reader or buffer.
         * @function decode
         * @memberof game.Size
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {game.Size} Size
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        Size.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            var end = length === undefined ? reader.len : reader.pos + length, message = new $root.game.Size();
            while (reader.pos < end) {
                var tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.width = reader.int32();
                    break;
                case 2:
                    message.height = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a Size message from the specified reader or buffer, length delimited.
         * @function decodeDelimited
         * @memberof game.Size
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {game.Size} Size
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        Size.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a Size message.
         * @function verify
         * @memberof game.Size
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        Size.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.width != null && message.hasOwnProperty("width"))
                if (!$util.isInteger(message.width))
                    return "width: integer expected";
            if (message.height != null && message.hasOwnProperty("height"))
                if (!$util.isInteger(message.height))
                    return "height: integer expected";
            return null;
        };

        /**
         * Creates a Size message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof game.Size
         * @static
         * @param {Object.<string,*>} object Plain object
         * @returns {game.Size} Size
         */
        Size.fromObject = function fromObject(object) {
            if (object instanceof $root.game.Size)
                return object;
            var message = new $root.game.Size();
            if (object.width != null)
                message.width = object.width | 0;
            if (object.height != null)
                message.height = object.height | 0;
            return message;
        };

        /**
         * Creates a plain object from a Size message. Also converts values to other types if specified.
         * @function toObject
         * @memberof game.Size
         * @static
         * @param {game.Size} message Size
         * @param {$protobuf.IConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        Size.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            var object = {};
            if (options.defaults) {
                object.width = 0;
                object.height = 0;
            }
            if (message.width != null && message.hasOwnProperty("width"))
                object.width = message.width;
            if (message.height != null && message.hasOwnProperty("height"))
                object.height = message.height;
            return object;
        };

        /**
         * Converts this Size to JSON.
         * @function toJSON
         * @memberof game.Size
         * @instance
         * @returns {Object.<string,*>} JSON object
         */
        Size.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return Size;
    })();

    game.GameState = (function() {

        /**
         * Properties of a GameState.
         * @memberof game
         * @interface IGameState
         * @property {Array.<game.IPoint>|null} [walls] GameState walls
         * @property {Array.<game.IAgent>|null} [agents] GameState agents
         * @property {game.ISize|null} [mapsize] GameState mapsize
         */

        /**
         * Constructs a new GameState.
         * @memberof game
         * @classdesc Represents a GameState.
         * @implements IGameState
         * @constructor
         * @param {game.IGameState=} [properties] Properties to set
         */
        function GameState(properties) {
            this.walls = [];
            this.agents = [];
            if (properties)
                for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * GameState walls.
         * @member {Array.<game.IPoint>} walls
         * @memberof game.GameState
         * @instance
         */
        GameState.prototype.walls = $util.emptyArray;

        /**
         * GameState agents.
         * @member {Array.<game.IAgent>} agents
         * @memberof game.GameState
         * @instance
         */
        GameState.prototype.agents = $util.emptyArray;

        /**
         * GameState mapsize.
         * @member {game.ISize|null|undefined} mapsize
         * @memberof game.GameState
         * @instance
         */
        GameState.prototype.mapsize = null;

        /**
         * Creates a new GameState instance using the specified properties.
         * @function create
         * @memberof game.GameState
         * @static
         * @param {game.IGameState=} [properties] Properties to set
         * @returns {game.GameState} GameState instance
         */
        GameState.create = function create(properties) {
            return new GameState(properties);
        };

        /**
         * Encodes the specified GameState message. Does not implicitly {@link game.GameState.verify|verify} messages.
         * @function encode
         * @memberof game.GameState
         * @static
         * @param {game.IGameState} message GameState message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GameState.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.walls != null && message.walls.length)
                for (var i = 0; i < message.walls.length; ++i)
                    $root.game.Point.encode(message.walls[i], writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            if (message.agents != null && message.agents.length)
                for (var i = 0; i < message.agents.length; ++i)
                    $root.game.Agent.encode(message.agents[i], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
            if (message.mapsize != null && Object.hasOwnProperty.call(message, "mapsize"))
                $root.game.Size.encode(message.mapsize, writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified GameState message, length delimited. Does not implicitly {@link game.GameState.verify|verify} messages.
         * @function encodeDelimited
         * @memberof game.GameState
         * @static
         * @param {game.IGameState} message GameState message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GameState.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a GameState message from the specified reader or buffer.
         * @function decode
         * @memberof game.GameState
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {game.GameState} GameState
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        GameState.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            var end = length === undefined ? reader.len : reader.pos + length, message = new $root.game.GameState();
            while (reader.pos < end) {
                var tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    if (!(message.walls && message.walls.length))
                        message.walls = [];
                    message.walls.push($root.game.Point.decode(reader, reader.uint32()));
                    break;
                case 2:
                    if (!(message.agents && message.agents.length))
                        message.agents = [];
                    message.agents.push($root.game.Agent.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.mapsize = $root.game.Size.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a GameState message from the specified reader or buffer, length delimited.
         * @function decodeDelimited
         * @memberof game.GameState
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {game.GameState} GameState
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        GameState.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a GameState message.
         * @function verify
         * @memberof game.GameState
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        GameState.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.walls != null && message.hasOwnProperty("walls")) {
                if (!Array.isArray(message.walls))
                    return "walls: array expected";
                for (var i = 0; i < message.walls.length; ++i) {
                    var error = $root.game.Point.verify(message.walls[i]);
                    if (error)
                        return "walls." + error;
                }
            }
            if (message.agents != null && message.hasOwnProperty("agents")) {
                if (!Array.isArray(message.agents))
                    return "agents: array expected";
                for (var i = 0; i < message.agents.length; ++i) {
                    var error = $root.game.Agent.verify(message.agents[i]);
                    if (error)
                        return "agents." + error;
                }
            }
            if (message.mapsize != null && message.hasOwnProperty("mapsize")) {
                var error = $root.game.Size.verify(message.mapsize);
                if (error)
                    return "mapsize." + error;
            }
            return null;
        };

        /**
         * Creates a GameState message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof game.GameState
         * @static
         * @param {Object.<string,*>} object Plain object
         * @returns {game.GameState} GameState
         */
        GameState.fromObject = function fromObject(object) {
            if (object instanceof $root.game.GameState)
                return object;
            var message = new $root.game.GameState();
            if (object.walls) {
                if (!Array.isArray(object.walls))
                    throw TypeError(".game.GameState.walls: array expected");
                message.walls = [];
                for (var i = 0; i < object.walls.length; ++i) {
                    if (typeof object.walls[i] !== "object")
                        throw TypeError(".game.GameState.walls: object expected");
                    message.walls[i] = $root.game.Point.fromObject(object.walls[i]);
                }
            }
            if (object.agents) {
                if (!Array.isArray(object.agents))
                    throw TypeError(".game.GameState.agents: array expected");
                message.agents = [];
                for (var i = 0; i < object.agents.length; ++i) {
                    if (typeof object.agents[i] !== "object")
                        throw TypeError(".game.GameState.agents: object expected");
                    message.agents[i] = $root.game.Agent.fromObject(object.agents[i]);
                }
            }
            if (object.mapsize != null) {
                if (typeof object.mapsize !== "object")
                    throw TypeError(".game.GameState.mapsize: object expected");
                message.mapsize = $root.game.Size.fromObject(object.mapsize);
            }
            return message;
        };

        /**
         * Creates a plain object from a GameState message. Also converts values to other types if specified.
         * @function toObject
         * @memberof game.GameState
         * @static
         * @param {game.GameState} message GameState
         * @param {$protobuf.IConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        GameState.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            var object = {};
            if (options.arrays || options.defaults) {
                object.walls = [];
                object.agents = [];
            }
            if (options.defaults)
                object.mapsize = null;
            if (message.walls && message.walls.length) {
                object.walls = [];
                for (var j = 0; j < message.walls.length; ++j)
                    object.walls[j] = $root.game.Point.toObject(message.walls[j], options);
            }
            if (message.agents && message.agents.length) {
                object.agents = [];
                for (var j = 0; j < message.agents.length; ++j)
                    object.agents[j] = $root.game.Agent.toObject(message.agents[j], options);
            }
            if (message.mapsize != null && message.hasOwnProperty("mapsize"))
                object.mapsize = $root.game.Size.toObject(message.mapsize, options);
            return object;
        };

        /**
         * Converts this GameState to JSON.
         * @function toJSON
         * @memberof game.GameState
         * @instance
         * @returns {Object.<string,*>} JSON object
         */
        GameState.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return GameState;
    })();

    return game;
})();

module.exports = $root;
